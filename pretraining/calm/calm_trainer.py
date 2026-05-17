# calm_trainer.py
#
# Trainer for CALM.  Mirrors LayerSkipTrainer:
#   - Logs final_ce_loss, per-exit exit_ce_loss_{l}, total_loss to wandb.
#   - Uses get_param_groups from multihead_trainer (no-decay for norms).
#   - No current_step injection (CALM has no step-dependent schedule).

import torch
import torch.distributed as dist
from transformers import Trainer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Nvium"))
from Nvium_trainer import get_param_groups


class CalmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_metrics_buffer = []
        self._eval_metrics_buffer = []

    # ------------------------------------------------------------------
    def training_step(self, model, inputs, *args, **kwargs):
        # Inject global_step so the model can apply the exit-loss warmup ramp.
        underlying = model.module if hasattr(model, "module") else model
        underlying.current_step = self.state.global_step
        return super().training_step(model, inputs, *args, **kwargs)

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        if hasattr(outputs, "loss_dict") and outputs.loss_dict is not None:
            metrics = {}
            for key, val in outputs.loss_dict.items():
                if isinstance(val, torch.Tensor):
                    metrics[key] = val.detach().item()
                else:
                    metrics[key] = val
            buf = (
                self._eval_metrics_buffer
                if getattr(self, "_in_eval", False)
                else self._train_metrics_buffer
            )
            buf.append(metrics)

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    # ------------------------------------------------------------------
    def _average_buffer(self, buffer):
        if not buffer:
            return {}
        all_keys = set().union(*(d.keys() for d in buffer))
        local_avgs = {}
        for k in all_keys:
            vals = [m[k] for m in buffer if k in m and isinstance(m[k], (int, float))]
            if vals:
                local_avgs[k] = sum(vals) / len(vals)
        if dist.is_initialized():
            with torch.no_grad():
                sorted_keys = sorted(local_avgs.keys())
                if sorted_keys:
                    t = torch.tensor(
                        [local_avgs[k] for k in sorted_keys],
                        device=self.model.device,
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    t /= dist.get_world_size()
                    local_avgs.update(dict(zip(sorted_keys, t.tolist())))
        return local_avgs

    # ------------------------------------------------------------------
    def log(self, logs: dict, *args, **kwargs) -> None:
        is_eval = any(k.startswith("eval") for k in logs)
        mode = "eval" if is_eval else "train"

        custom_metrics = self._average_buffer(
            self._eval_metrics_buffer if is_eval else self._train_metrics_buffer
        )
        if is_eval:
            self._eval_metrics_buffer = []
        else:
            self._train_metrics_buffer = []

        if not self.is_world_process_zero():
            return

        final_logs = {}
        for k, v in logs.items():
            clean_k = k.replace("train/", "").replace("eval_", "").replace("eval/", "")
            final_logs[
                f"train/{clean_k}"
                if ("loss" in clean_k or "lr" in clean_k)
                else clean_k
            ] = v
        for k, v in custom_metrics.items():
            final_logs[f"{mode}/{k}"] = v

        super().log(final_logs, *args, **kwargs)

    # ------------------------------------------------------------------
    def evaluate(self, *args, **kwargs):
        self._in_eval = True
        self._eval_metrics_buffer = []
        try:
            result = super().evaluate(*args, **kwargs)
        finally:
            self._in_eval = False
        return result

    # ------------------------------------------------------------------
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs.loss.detach() if outputs.loss is not None else None
        if hasattr(outputs, "loss_dict") and outputs.loss_dict is not None:
            metrics = {
                k: v.detach().item() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.loss_dict.items()
            }
            if getattr(self, "_in_eval", False):
                self._eval_metrics_buffer.append(metrics)
        return (loss, None, None)

    # ------------------------------------------------------------------
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
