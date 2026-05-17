# calm_model.py
#
# CALM: Confident Adaptive Language Modelling
# Schuster et al., NeurIPS 2022 (arXiv:2207.07061)
#
# -----------------------------------------------------------------------
# Training recipe (for fair comparison with baseline / LayerSkip)
# -----------------------------------------------------------------------
#
# 1. Start from a fully trained dense baseline checkpoint (warm start).
# 2. Fine-tune the FULL model (backbone unfrozen) with:
#      loss = CE(final) + Σ_l CE(exit_l)      [equal weights]
#    using the SHARED model.norm + lm_head at every exit — zero extra
#    parameters, same as LayerSkip.
# 3. Fine-tune for 25% of the original training budget (4321 steps) with
#    a fresh cosine schedule starting at the original min_lr (4e-5 → 4e-6).
#
# This is directly comparable to "continued baseline" which runs the same
# architecture for the same extra steps but with only the final CE loss.
#
# Inference:
#   At each exit layer, compute confidence = max(softmax(logits)).
#   Exit early if confidence > threshold τ (swept on validation set).
# -----------------------------------------------------------------------

import copy
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from typing import Optional, List
from dataclasses import dataclass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


@dataclass
class CalmOutput(CausalLMOutputWithPast):
    loss_dict: Optional[dict] = None


class CalmModel(LlamaForCausalLM):
    """
    CALM wrapper around LlamaForCausalLM.

    Zero extra parameters: uses the shared model.norm + lm_head at every
    exit, identical to LayerSkip.  The full model (backbone included) is
    fine-tuned from the dense baseline checkpoint.

    Training forward: CE at each exit layer + CE at final layer.
    Eval forward (forward_adaptive): simulate per-token adaptive exit for
    the threshold sweep; used only for perplexity measurement.
    """

    def __init__(self, config, exp_config=None, **kwargs):
        attn_impl = kwargs.pop("attn_implementation", None)
        super().__init__(config, **kwargs)

        if exp_config is None:
            raise ValueError("exp_config must be provided")

        self.cfg = exp_config
        if attn_impl is not None:
            self.config._attn_implementation = attn_impl

        # Exit layer indices (0-indexed)
        exit_layers_cfg = self.cfg.get("early_layer_idx", [])
        if isinstance(exit_layers_cfg, list):
            self.exit_layer_indices = sorted(int(l) for l in exit_layers_cfg)
        else:
            self.exit_layer_indices = [int(exit_layers_cfg)]
        self._exit_set = set(self.exit_layer_indices)

        # Inference threshold (default; swept in calm_eval.py)
        calm_cfg = self.cfg.get("calm", {})
        self.confidence_threshold = float(calm_cfg.get("threshold", 0.9))

        # ---------------------------------------------------------------- #
        # Optional per-exit RMSNorms (deep-copied from model.norm).
        # CALM paper (T5) shares the final norm at every exit.
        # For Llama (RMSNorm) backbones, hidden-state magnitudes vary by
        # depth, so per-exit norms are a safer default for shallow exits.
        # Weights are re-synced from the loaded `model.norm` in calm_main
        # after the baseline state_dict is loaded.
        # ---------------------------------------------------------------- #
        self.per_exit_norms_enabled = bool(calm_cfg.get("per_exit_norms", False))
        if self.per_exit_norms_enabled:
            self.exit_norms = nn.ModuleDict({
                str(i): copy.deepcopy(self.model.norm)
                for i in self.exit_layer_indices
            })

        # ---------------------------------------------------------------- #
        # Per-layer loss weighting (Appendix D of CALM).
        #   linear:  w_l = (l+1) / Σ_{l' ∈ exits ∪ final} (l'+1)
        #   equal :  w_l = 1 / (|exits| + 1)           (legacy behaviour)
        # ---------------------------------------------------------------- #
        self.loss_weighting = str(calm_cfg.get("loss_weighting", "linear")).lower()
        if self.loss_weighting not in ("linear", "equal"):
            raise ValueError(
                f"calm.loss_weighting must be 'linear' or 'equal', got "
                f"{self.loss_weighting!r}"
            )

        # Pre-compute weights. All indices are 0-indexed.
        L = int(config.num_hidden_layers)
        final_idx = L - 1
        all_indices = list(self.exit_layer_indices) + [final_idx]
        if self.loss_weighting == "linear":
            raw = [idx + 1 for idx in all_indices]
            total = float(sum(raw))
            self._loss_weights = {idx: w / total for idx, w in zip(all_indices, raw)}
        else:  # equal
            w = 1.0 / len(all_indices)
            self._loss_weights = {idx: w for idx in all_indices}
        self._final_idx = final_idx

        # ---------------------------------------------------------------- #
        # Exit loss warmup: linearly ramp exit losses from 0 → full weight
        # over the first exit_loss_warmup_steps steps. Keeps early-training
        # stable: step 0 trains only on final CE, exits join gradually.
        # total_loss = (1 - alpha) * final_ce + alpha * calm_weighted_loss
        # ---------------------------------------------------------------- #
        self.exit_loss_warmup_steps = int(calm_cfg.get("exit_loss_warmup_steps", 0))
        self.current_step = 0

    # ---------------------------------------------------------------------- #
    # Internal: layer loop collecting exit logits
    # ---------------------------------------------------------------------- #

    def _run_layers(self, inputs_embeds, causal_mask, position_ids,
                    position_embeddings, past_key_values, cache_position,
                    use_cache, collect_exit_logits=False):
        hidden_states = inputs_embeds
        exit_logits_map = {}

        for i, layer in enumerate(self.model.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            if collect_exit_logits and (i + 1 in self._exit_set):
                # Shared lm_head. Norm is either shared model.norm (paper) or
                # a per-exit RMSNorm (Llama-friendly; opt-in via config).
                # Key is i+1 to match config indices (exit_layer_idx convention:
                # index N = exit after N layers processed, matching multihead model).
                exit_idx = i + 1
                if self.per_exit_norms_enabled:
                    normed = self.exit_norms[str(exit_idx)](hidden_states)
                else:
                    normed = self.model.norm(hidden_states)
                exit_logits_map[exit_idx] = self.lm_head(normed)

        return hidden_states, exit_logits_map

    # ---------------------------------------------------------------------- #
    # Training forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            raise ValueError("Must provide input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = DynamicCache()
        past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, device=device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        if use_cache is None:
            use_cache = not self.training

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        causal_mask = create_causal_mask(
            self.config, inputs_embeds, attention_mask,
            cache_position, past_key_values, position_ids,
        )
        position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)

        collect = (labels is not None)
        hidden_states, exit_logits_map = self._run_layers(
            inputs_embeds, causal_mask, position_ids, position_embeddings,
            past_key_values, cache_position, use_cache,
            collect_exit_logits=collect,
        )

        final_hidden = self.model.norm(hidden_states)
        logits = self.lm_head(final_hidden)

        total_loss = None
        loss_dict = {}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_labels = labels[..., 1:].contiguous()

            # Final layer CE
            shift_logits = logits[..., :-1, :].contiguous()
            final_ce = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss_dict["final_ce_loss"] = final_ce

            # Per-exit CE (unweighted, for logging)
            exit_ces = {}
            for l, exit_log in exit_logits_map.items():
                shift_exit = exit_log[..., :-1, :].contiguous()
                ce = loss_fct(
                    shift_exit.view(-1, shift_exit.size(-1)),
                    shift_labels.view(-1),
                )
                exit_ces[l] = ce
                loss_dict[f"exit_ce_loss_{l}"] = ce

            # Warmup ramp: alpha=0 → final CE only; alpha=1 → full CALM loss
            if self.exit_loss_warmup_steps > 0:
                alpha = min(1.0, float(self.current_step) / float(self.exit_loss_warmup_steps))
            else:
                alpha = 1.0

            # Weighted aggregate (paper Appendix D: linear-by-layer-index)
            # total_loss = (1-alpha)*final_ce + alpha * Σ w_l*ce_l
            calm_weighted = float(self._loss_weights[self._final_idx]) * final_ce
            for l, ce in exit_ces.items():
                calm_weighted = calm_weighted + float(self._loss_weights[l]) * ce
            total_loss = (1.0 - alpha) * final_ce + alpha * calm_weighted

            loss_dict["total_loss"] = total_loss
            loss_dict["exit_loss_alpha"] = alpha
            loss_dict["n_exit_losses"] = float(len(exit_logits_map))
            loss_dict["loss_weighting"] = self.loss_weighting

        return CalmOutput(
            loss=total_loss,
            logits=logits,
            past_key_values=past_key_values,
            loss_dict=loss_dict if loss_dict else None,
        )

    # ---------------------------------------------------------------------- #
    # Perplexity evaluation with adaptive exit (threshold sweep)
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def forward_adaptive(self, input_ids, labels, threshold):
        """
        Simulate CALM's per-token adaptive early exit on full sequences.

        Runs all layers once, collects logits at each exit point, then
        per (batch, position) picks the earliest exit where
        max(softmax(logits)) > threshold.  Final layer is the fallback.

        NOTE: this is a measurement tool — it runs all layers so there is
        no actual compute saving here.  Use it for the CE vs. exit-depth
        Pareto curve.  Actual generation speedup requires a fixed exit
        layer with a proper KV cache (see forward_fixed_exit).

        Returns:
            ce:              scalar — mean CE using adaptive-exit predictions
            avg_exit_layer:  float  — mean chosen exit layer index
            early_exit_frac: float  — fraction of tokens that exited early
        """
        device = input_ids.device
        B, S = input_ids.shape
        L = self.config.num_hidden_layers

        cache_position = torch.arange(S, device=device)
        position_ids = cache_position.unsqueeze(0).expand(B, -1)
        past_key_values = DynamicCache()

        inputs_embeds = self.model.embed_tokens(input_ids)
        causal_mask = create_causal_mask(
            self.config, inputs_embeds, None,
            cache_position, past_key_values, position_ids,
        )
        position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)

        hidden_states, exit_logits_map = self._run_layers(
            inputs_embeds, causal_mask, position_ids, position_embeddings,
            past_key_values, cache_position, use_cache=False,
            collect_exit_logits=True,
        )

        # Add final layer
        final_hidden = self.model.norm(hidden_states)
        exit_logits_map[L - 1] = self.lm_head(final_hidden)

        sorted_exits = sorted(exit_logits_map.keys())
        n_exits = len(sorted_exits)
        V = exit_logits_map[sorted_exits[0]].size(-1)
        S_pred = S - 1

        shift_labels = labels[..., 1:].contiguous()  # [B, S_pred]

        # Stack: [n_exits, B, S_pred, V]
        all_logits = torch.stack(
            [exit_logits_map[l][..., :-1, :].float() for l in sorted_exits], dim=0
        )

        # Confidence: [n_exits, B, S_pred]
        confidence = torch.softmax(all_logits, dim=-1).max(dim=-1).values

        # Per-token: first exit index where confidence > threshold
        exits_above = confidence > threshold          # [n_exits, B, S_pred]
        has_exit = exits_above.any(dim=0)             # [B, S_pred]
        first_exit_idx = exits_above.long().argmax(dim=0)
        final_idx = n_exits - 1
        chosen_idx = torch.where(
            has_exit, first_exit_idx,
            torch.full_like(first_exit_idx, final_idx),
        )  # [B, S_pred]

        # Gather logits from chosen exit: [B, S_pred, V]
        all_logits_perm = all_logits.permute(1, 2, 0, 3)          # [B, S_pred, n_exits, V]
        idx_gather = chosen_idx.unsqueeze(-1).unsqueeze(-1).expand(B, S_pred, 1, V)
        chosen_logits = all_logits_perm.gather(2, idx_gather).squeeze(2)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        ce = loss_fct(chosen_logits.reshape(-1, V), shift_labels.reshape(-1))

        sorted_exits_t = torch.tensor(sorted_exits, device=device, dtype=torch.float)
        avg_exit_layer = sorted_exits_t[chosen_idx].mean().item()
        early_exit_frac = (chosen_idx < final_idx).float().mean().item()

        return ce, avg_exit_layer, early_exit_frac

    # ---------------------------------------------------------------------- #
    # Parameter summary
    # ---------------------------------------------------------------------- #

    def print_parameter_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        extra = 0
        if self.per_exit_norms_enabled:
            extra = sum(p.numel() for p in self.exit_norms.parameters())
        print(f"[CALM] Total params:       {total/1e6:.2f}M")
        print(f"[CALM] Trainable params:   {trainable/1e6:.2f}M")
        print(f"[CALM] Extra params:       {extra}  "
              f"({'per-exit RMSNorms' if extra else 'shared model.norm + lm_head'})")
        print(f"[CALM] Exit layers ({len(self.exit_layer_indices)}):  {self.exit_layer_indices}")
        print(f"[CALM] Loss weighting:     {self.loss_weighting}")
        w_str = ", ".join(
            f"L{idx}:{self._loss_weights[idx]:.3f}"
            for idx in sorted(self._loss_weights)
        )
        print(f"[CALM] Weights:            {w_str}")
        print(f"[CALM] Default threshold:  {self.confidence_threshold}")
