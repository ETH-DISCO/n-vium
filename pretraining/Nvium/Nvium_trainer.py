# Nvium_trainer.py
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import numpy as np
import wandb
from transformers import Trainer, TrainerCallback
import torch.distributed as dist


def get_param_groups(model, weight_decay, base_lr=None, router_lr_multiplier=1.0, return_debug=False):
    no_decay_substrings = ["bias", "norm", "layernorm", "layer_norm", "rmsnorm", "ln"]

    decay, no_decay, router = [], [], []
    decay_names, no_decay_names, router_names = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()

        is_no_decay = any(s in n for s in no_decay_substrings)
        if "embed_tokens" in n:
            is_no_decay = True

        if is_no_decay:
            no_decay.append(p)
            no_decay_names.append(name)
        else:
            decay.append(p)
            decay_names.append(name)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if router:
        router_lr = base_lr * router_lr_multiplier if base_lr else None
        groups.append({
            "params": router,
            "weight_decay": 0.0,
            **({"lr": router_lr} if router_lr else {}),
        })

    if return_debug:
        dbg = {
            f"DECAY={weight_decay}": decay_names,
            "NO_DECAY=0.0": no_decay_names,
            f"ROUTER (lr×{router_lr_multiplier})": router_names,
        }
        return groups, dbg

    return groups


class ForwardTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_metrics_buffer = []
        self._eval_metrics_buffer = []
        self.histogram_logging_steps = 100
        
        # Histogram Bins (0 to 1 in 10 steps)
        self.bins = np.linspace(0.0, 1.0, 11)
        self.bin_labels = [f"{round(self.bins[i],1)}-{round(self.bins[i+1],1)}" for i in range(10)]
        
        # Buffers for running mean of histograms (one per router, lazily initialised)
        self.hist_running_sums = {}  # {router_idx: np.ndarray}
        self.hist_counts = {}        # {router_idx: int}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pass current step so model can switch warmup/main phases
        inputs["current_step"] = self.state.global_step

        outputs = model(**inputs)
        if hasattr(outputs, "loss_dict") and outputs.loss_dict is not None:
            metrics = {}
            
            # DETECT MODEL TYPE: Single-exit vs Multi-exit vs Baseline (None)
            has_router = getattr(outputs, "router_weights", None) is not None
            is_Nvium = has_router and isinstance(outputs.router_weights, list)

            if not has_router:
                pass  # Baseline model: no router metrics to log
            elif is_Nvium:
                # Nvium
                # router_weights is list of tensors [B, L, 2] where 2 = [P(exit), P(continue)]
                # exit_probabilities is tensor [B, L, num_exits] with total exit probabilities
                
                num_routers = len(outputs.router_weights)

                # Per-router: conditional probabilities for means, total marginal for histograms
                for i, router_weights in enumerate(outputs.router_weights):
                    p_exit_cond = router_weights[..., 0]  # P(exit | reached here)
                    p_continue = router_weights[..., 1]

                    metrics[f"router_{i+1}_exit_mean"] = p_exit_cond.mean().detach().item()
                    metrics[f"router_{i+1}_continue_mean"] = p_continue.mean().detach().item()

                    w_vals = p_exit_cond.detach().float().cpu().numpy().flatten()
                    counts, _ = np.histogram(w_vals, bins=self.bins)
                    metrics[f"hist_counts_{i}"] = counts

            else:
                # Bivium (backward compatibility)
                w1 = outputs.router_weights[..., 0]  # [B, L]
                
                metrics["early_w1_mean"] = w1.mean().detach().item()
                metrics["final_w2_mean"] = (1.0 - w1).mean().detach().item()
                
                # Histogram
                w1_vals = w1.detach().float().cpu().numpy().flatten()
                counts, _ = np.histogram(w1_vals, bins=self.bins)
                metrics["w1_hist_counts"] = counts

            # 4. Dynamically add everything from the model's loss_dict
            for key, val in outputs.loss_dict.items():
                if isinstance(val, torch.Tensor):
                    metrics[key] = val.detach().item()
                else:
                    metrics[key] = val
            
            if getattr(self, "_in_eval", False):
                self._eval_metrics_buffer.append(metrics)
            else:
                self._train_metrics_buffer.append(metrics)
        
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _average_buffer(self, buffer):
        
        if not buffer: 
            return {}
            
        # 1. Identify all unique keys; separate histogram keys from scalar keys
        all_keys = set().union(*(d.keys() for d in buffer))
        hist_keys = sorted(k for k in all_keys if k.startswith("hist_counts_"))
        scalar_keys = [k for k in all_keys if k not in hist_keys]

        # 2. Average scalars
        local_avgs = {}
        for k in scalar_keys:
            values = [m[k] for m in buffer if k in m]
            if values:
                local_avgs[k] = sum(values) / len(values)

        # 3. Sum histograms per router
        local_hists = {}
        for hk in hist_keys:
            s = sum(m[hk] for m in buffer if hk in m)
            local_hists[hk] = s if not isinstance(s, int) else np.zeros(len(self.bins) - 1)

        if dist.is_initialized():
            with torch.no_grad():
                sorted_keys = sorted(local_avgs.keys())
                if sorted_keys:
                    metrics_tensor = torch.tensor([local_avgs[k] for k in sorted_keys], device=self.model.device)
                    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                    metrics_tensor /= dist.get_world_size()
                    local_avgs.update(dict(zip(sorted_keys, metrics_tensor.tolist())))

                for hk, hist in local_hists.items():
                    hist_tensor = torch.tensor(hist, device=self.model.device, dtype=torch.float32)
                    dist.all_reduce(hist_tensor, op=dist.ReduceOp.SUM)
                    local_avgs[hk] = hist_tensor.cpu().numpy()
                return local_avgs

        local_avgs.update(local_hists)
        return local_avgs

    def log(self, logs: dict, *args, **kwargs) -> None:
        is_eval = any(k.startswith("eval") for k in logs)
        mode = "eval" if is_eval else "train"
        
        custom_metrics = self._average_buffer(
            self._eval_metrics_buffer if is_eval else self._train_metrics_buffer
        )
        if is_eval:
            self._eval_metrics_buffer = []
            # Cache averaged eval metrics so callbacks (BaselineCESaver,
            # AdaptivePenalty) can read raw keys like "final_ce_loss" / "mix_loss"
            # before they get namespaced into "eval/..." below.
            self._last_eval_metrics = custom_metrics
        else:
            self._train_metrics_buffer = []

        if not self.is_world_process_zero():
            return

        final_logs = {}
        
        # Clean standard logs
        for k, v in logs.items():
            clean_k = k.replace("train/", "").replace("eval_", "").replace("eval/", "")
            final_logs[f"train/{clean_k}" if "loss" in clean_k or "lr" in clean_k else clean_k] = v

        # Histogram logging (every 100 steps) — one per router
        # Logged directly via wandb.log() to avoid JSON serialization crash in Trainer state
        hist_keys = sorted(k for k in list(custom_metrics.keys()) if k.startswith("hist_counts_"))
        wandb_hist_logs = {}
        for hk in hist_keys:
            counts = custom_metrics.pop(hk)
            router_idx = int(hk.split("_")[-1])
            total_tokens = counts.sum()
            if total_tokens > 0:
                prop = counts / total_tokens
                if router_idx not in self.hist_running_sums:
                    self.hist_running_sums[router_idx] = np.zeros(len(self.bins) - 1, dtype=np.float64)
                    self.hist_counts[router_idx] = 0
                self.hist_running_sums[router_idx] += prop
                self.hist_counts[router_idx] += 1

                if self.state.global_step % self.histogram_logging_steps == 0:
                    mean_prop = self.hist_running_sums[router_idx] / self.hist_counts[router_idx]
                    wandb_hist_logs[f"exit_hist/router_{router_idx + 1}/step"] = wandb.Histogram(
                        np_histogram=(prop, self.bins)
                    )
                    wandb_hist_logs[f"exit_hist/router_{router_idx + 1}/mean"] = wandb.Histogram(
                        np_histogram=(mean_prop, self.bins)
                    )
        if wandb_hist_logs and wandb.run is not None:
            wandb.log(wandb_hist_logs, step=self.state.global_step)

        # Map remaining custom scalars
        for k, v in custom_metrics.items():
            # CASE A: Head metrics (distribution comparisons)
            if k.startswith("heads/"):
                pass

            # CASE B: Pre-grouped metrics (contain "/") — prepend mode only
            elif "/" in k:
                final_logs[f"{mode}/{k}"] = v

            # CASE C: Router metrics (legacy flat keys)
            elif "router" in k or "exit" in k or "mean" in k:
                final_logs[f"router/{mode}/{k}"] = v

            # CASE D: Everything else (loss components, etc.)
            else:
                final_logs[f"{mode}/{k}"] = v

        # Expose the (possibly adaptive) compute-penalty weight every log call
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "alpha_map_main"):
            final_logs["train/compute_penalty_weight"] = model.alpha_map_main.get("router_compute_loss", 0.0)

        super().log(final_logs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self._in_eval = True
        self._eval_metrics_buffer = []

        try:
            if hasattr(self.model, "_orig_mod"):
                compiled = self.model
                orig = compiled._orig_mod

                # Swap both references the Trainer uses
                self.model = orig
                self.model_wrapped = orig

                result = super().evaluate(*args, **kwargs)

                self.model = compiled
                self.model_wrapped = compiled
            else:
                result = super().evaluate(*args, **kwargs)
        finally:
            self._in_eval = False

        return result

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        inputs["current_step"] = self.state.global_step

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss.detach() if outputs.loss is not None else None

        # Fill eval buffer
        if hasattr(outputs, "loss_dict") and outputs.loss_dict is not None:
            metrics = {}
            for key, val in outputs.loss_dict.items():
                if isinstance(val, torch.Tensor):
                    metrics[key] = val.detach().item()
                else:
                    metrics[key] = val

            if getattr(outputs, "router_weights", None) is not None:
                is_multi_exit = isinstance(outputs.router_weights, list)

                if is_multi_exit:
                    for i, router_weights in enumerate(outputs.router_weights):
                        p_exit_cond = router_weights[..., 0]
                        metrics[f"router_{i+1}_exit_mean"] = p_exit_cond.mean().detach().item()
                        metrics[f"router_{i+1}_continue_mean"] = router_weights[..., 1].mean().detach().item()

                        w_vals = p_exit_cond.detach().float().cpu().numpy().flatten()
                        counts, _ = np.histogram(w_vals, bins=self.bins)
                        metrics[f"hist_counts_{i}"] = counts
                else:
                    w1 = outputs.router_weights[..., 0]
                    metrics["early_w1_mean"] = w1.mean().detach().item()
                    metrics["final_w2_mean"] = (1.0 - w1).mean().detach().item()
                    w1_vals = w1.detach().float().cpu().numpy().flatten()
                    counts, _ = np.histogram(w1_vals, bins=self.bins)
                    metrics["hist_counts_0"] = counts

            if getattr(self, "_in_eval", False):
                self._eval_metrics_buffer.append(metrics)

        return (loss, None, None)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        wd = self.args.weight_decay
        param_groups, dbg = get_param_groups(
            self.model, wd,
            base_lr=self.args.learning_rate,
            router_lr_multiplier=1.0,
            return_debug=True,
        )

        use_fused = "fused" in self.args.optim if self.args.optim else False

        if self.is_world_process_zero():
            print(f"[OPTIMIZER] Using fused AdamW: {use_fused}")
            for group_name, names in dbg.items():
                print(f"[OPTIMIZER] {group_name}: {names}")

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            fused=use_fused,
        )
        return self.optimizer


class BaselineCESaverCallback(TrainerCallback):
    """Saves {global_step: final_ce_loss} from eval runs to a JSON file.
    Attach to baseline runs so Nvium runs can load the curve as a quality target.
    Set cb._trainer_ref = trainer after construction so the callback can access
    ForwardTrainer._last_eval_metrics."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.records: dict = {}
        self._trainer_ref = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        # on_log fires after ForwardTrainer.log(), which sets _last_eval_metrics.
        if logs is None or not any(k.startswith("eval") for k in logs):
            return
        if self._trainer_ref is None or not self._trainer_ref.is_world_process_zero():
            return
        last = getattr(self._trainer_ref, "_last_eval_metrics", {})
        final_ce = last.get("final_ce_loss")
        if final_ce is not None:
            self.records[str(state.global_step)] = final_ce
            with open(self.output_path, "w") as f:
                json.dump(self.records, f, indent=2)


class AdaptivePenaltyCallback(TrainerCallback):
    """Adjusts router_compute_loss weight after each eval checkpoint.

    Uses an asymmetric proportional controller on the quality gap:
        gap = baseline_ce - mix_ce
    Positive gap → mix_ce is better than baseline → increase penalty (k_up).
    Negative gap → mix_ce is worse than baseline  → decrease penalty (k_down).
    A dead_band suppresses adjustments when |gap| is within eval noise.
    max_step caps the per-checkpoint weight delta to prevent overshoot.
    """

    def __init__(self, model, baseline_ce_path: str,
                 k_up: float = 2.0, k_down: float = 4.0,
                 dead_band: float = 0.005, max_step: float = 0.2,
                 min_weight: float = 0.0, max_weight: float = 2.0):
        self._trainer_ref = None
        self.model = model
        self.baseline_ce_path = baseline_ce_path
        self.k_up = k_up
        self.k_down = k_down
        self.dead_band = dead_band
        self.max_step = max_step
        self.min_weight = min_weight
        self.max_weight = max_weight

    def _nearest_baseline_ce(self, step: int) -> float:
        # Re-read from disk every eval so parallel baseline runs are always up to date.
        with open(self.baseline_ce_path) as f:
            records = json.load(f)
        baseline = {int(k): float(v) for k, v in records.items()}
        if not baseline:
            return float("inf")
        nearest = min(baseline.keys(), key=lambda s: abs(s - step))
        return baseline[nearest]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not any(k.startswith("eval") for k in logs):
            return

        trainer = getattr(self, "_trainer_ref", None)
        last = getattr(trainer, "_last_eval_metrics", {}) if trainer else {}

        mix_ce = last.get("mix_loss")
        if mix_ce is None:
            return

        target_ce = self._nearest_baseline_ce(state.global_step)
        current_weight = self.model.alpha_map_main.get("router_compute_loss", 0.15)
        gap = target_ce - mix_ce  # positive = headroom, negative = over-budget

        if abs(gap) < self.dead_band:
            direction = "hold"
            new_weight = current_weight
            delta = 0.0
        elif gap > 0:
            delta = min(self.k_up * gap, self.max_step)
            new_weight = min(current_weight + delta, self.max_weight)
            direction = "increase"
        else:
            delta = max(self.k_down * gap, -self.max_step)
            new_weight = max(current_weight + delta, self.min_weight)
            direction = "decrease"

        self.model.alpha_map_main["router_compute_loss"] = new_weight
        self.model.active_main["router_compute_loss"] = new_weight > 0.0

        if trainer is not None and trainer.is_world_process_zero():
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"[ADAPTIVE PENALTY] step {state.global_step}")
            print(f"  Baseline final_ce  : {target_ce:.6f}")
            print(f"  Nvium mix_ce       : {mix_ce:.6f}")
            print(f"  Gap (target-mix)   : {gap:+.6f}  ({'headroom' if gap > 0 else 'over-budget' if gap < 0 else 'at target'})")
            print(f"  Dead band          : ±{self.dead_band:.4f}")
            if abs(gap) < self.dead_band:
                print(f"  Decision           : HOLD  (|gap| < dead_band)")
            elif gap > 0:
                print(f"  Decision           : INCREASE  (k_up={self.k_up} × gap={gap:.4f} = {self.k_up*gap:.4f}, capped at max_step={self.max_step} → delta={delta:.4f})")
            else:
                print(f"  Decision           : DECREASE  (k_down={self.k_down} × gap={gap:.4f} = {self.k_down*gap:.4f}, capped at -max_step={-self.max_step} → delta={delta:.4f})")
            print(f"  Beta               : {current_weight:.6f} → {new_weight:.6f}")
            print(sep + "\n")

            if wandb.run is not None:
                wandb.log({
                    "adaptive/compute_penalty_weight": new_weight,
                    "adaptive/mix_ce": mix_ce,
                    "adaptive/baseline_ce_target": target_ce,
                    "adaptive/gap": gap,
                    "adaptive/delta": delta,
                    "adaptive/direction": direction,
                }, step=state.global_step)