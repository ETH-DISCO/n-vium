# layerskip_model.py
#
# LayerSkip (Elhoushi et al., Meta, arXiv:2404.16710 / ACL 2024)
#
# -----------------------------------------------------------------------
# Training recipe (corrected to match paper exactly)
# -----------------------------------------------------------------------
#
# 1. LAYER DROPOUT
#    p_{l,t} = S(t) * D(l) * p_max
#
#    D(l) = exp(ln2 * l / (L-1)) - 1     ← exponential ramp over layers
#                                            D(0)=0, D(L-1)=1
#
#    S(t) for continual pretraining / fine-tuning: S(t) = 1  (constant)
#    S(t) for from-scratch pretraining:            S(t) = exp(ln2 * t/T) - 1
#         This exponential curriculum ramps dropout from 0 to p_max as
#         training progresses, helping the model first learn without dropout.
#
# 2. EARLY EXIT LOSS
#    J = Σ_{l=0}^{L-1}  ẽ(t,l) * CE(lm_head(norm(h_l)), Y)
#
#    Weights ẽ(t,l) = C(t,l) * e(l)  (normalised to sum=1 over active layers)
#
#    e(l): per-layer scale. Deeper layers get MORE weight because they are
#          more capable. Paper uses e(l) ∝ (l+1)^2 (quadratic) * escale,
#          where escale is a global hyperparameter (default 0.1–0.2).
#
#    C(t,l): curriculum that gradually enables exit losses.
#      - "gradual"    (C_grad):  enable layer L-1 first, then L-2, …, 0
#                                one new layer every T/(2L) steps.
#                                Recommended for finetuning.
#      - "rotational" (C_rot,R): at each training step only one layer
#                                (rotating round-robin every R steps) is active.
#                                Recommended for continual pretraining.
#      - "all":        all layers active from step 0 (simple baseline).
#
#    The SAME shared lm_head (and model.norm) is used at every exit.
#    Zero extra parameters.
#
# -----------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from typing import Optional, List
from dataclasses import dataclass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Output dataclass — mirrors multihead CustomModelOutput for trainer compat
# ---------------------------------------------------------------------------

@dataclass
class LayerSkipOutput(CausalLMOutputWithPast):
    loss_dict: Optional[dict] = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LayerSkipModel(LlamaForCausalLM):
    """
    Standard LlamaForCausalLM with LayerSkip training augmentations:
      - Exponential layer dropout with optional time-based curriculum
      - Auxiliary early-exit CE loss through the SHARED lm_head (no extra params)
    """

    def __init__(self, config, exp_config=None, **kwargs):
        attn_impl = kwargs.pop("attn_implementation", None)
        super().__init__(config, **kwargs)

        if exp_config is None:
            raise ValueError("exp_config must be provided")

        self.cfg = exp_config

        if attn_impl is not None:
            self.config._attn_implementation = attn_impl

        L = config.num_hidden_layers

        # ------------------------------------------------------------------ #
        # LayerSkip config
        # ------------------------------------------------------------------ #
        ls_cfg = self.cfg.get("layerskip", {})

        # --- Layer dropout ---
        self.p_max = float(ls_cfg.get("p_max", 0.1))

        # D(l): exponential ramp, D(0)=0, D(L-1)=1
        # D(l) = exp(ln2 * l/(L-1)) - 1
        self._D = [
            math.exp(math.log(2) * l / max(L - 1, 1)) - 1.0
            for l in range(L)
        ]

        # S(t) schedule: "constant" or "exponential"
        # constant  → S(t)=1 (continual pretraining / fine-tuning)
        # exponential → S(t)=exp(ln2*t/T)-1 (from-scratch pretraining)
        self.dropout_schedule = ls_cfg.get("dropout_schedule", "exponential")
        self.total_training_steps = int(ls_cfg.get("total_training_steps", 1))

        # --- Early-exit loss ---
        # Which layers are active exit points.
        # Use all layers by default (full LayerSkip), or a subset.
        early_cfg = self.cfg.get("early_layer_idx", None)
        if early_cfg is None:
            self.early_layer_indices = list(range(L))
        elif isinstance(early_cfg, list):
            self.early_layer_indices = sorted(early_cfg)
        else:
            self.early_layer_indices = [int(early_cfg)]

        self._early_exit_set = set(self.early_layer_indices)

        # e(l): per-layer scale — quadratic, deeper gets more weight
        # e(l) = ((l+1)/L)^2 * escale
        self.escale = float(ls_cfg.get("escale", 0.1))
        self._e = {
            l: ((l + 1) / L) ** 2 * self.escale
            for l in self.early_layer_indices
        }

        # Curriculum type: "all" | "gradual" | "rotational"
        self.curriculum_type = ls_cfg.get("curriculum_type", "gradual")
        # For rotational curriculum: rotation period R (steps)
        self.curriculum_rotation_period = int(ls_cfg.get("curriculum_rotation_period", 8))

        # Temperature for CE (matches multihead)
        self.loss_temp = float(self.cfg.get("temperature", 1.0))

        # Dummy warmup_steps_limit so the trainer's step injection is a no-op
        self.warmup_steps_limit = 0

        # ------------------------------------------------------------------ #
        # Random init
        # ------------------------------------------------------------------ #
        if self.cfg.get("random_init_all", False):
            self._random_init_weights()

    # ---------------------------------------------------------------------- #
    # Curriculum helpers
    # ---------------------------------------------------------------------- #

    def _S(self, t: int) -> float:
        """Time-step scale for layer dropout."""
        if self.dropout_schedule == "constant":
            return 1.0
        # Exponential curriculum: S(t) = exp(ln2 * t/T) - 1, capped at 1
        T = max(self.total_training_steps, 1)
        return min(math.exp(math.log(2) * t / T) - 1.0, 1.0)

    def _active_exit_layers(self, current_step: int) -> List[int]:
        """
        Return which exit layers contribute loss at this training step,
        according to the curriculum.
        """
        if self.curriculum_type == "all":
            return self.early_layer_indices

        L = self.config.num_hidden_layers
        n = len(self.early_layer_indices)
        if n == 0:
            return []

        if self.curriculum_type == "gradual":
            # Enable one new layer (from deepest to shallowest) every T/(2L) steps.
            # At step 0 only layer L-1 is active; at step T/2 all layers are active.
            T = max(self.total_training_steps, 1)
            period = max(T // (2 * L), 1)
            # Number of layers enabled so far (count from the end of early_layer_indices)
            n_enabled = min(current_step // period + 1, n)
            # Enable from deepest index backwards
            return self.early_layer_indices[-n_enabled:]

        elif self.curriculum_type == "rotational":
            # Only one layer active at each step, rotating every R steps.
            R = self.curriculum_rotation_period
            idx = (current_step // R) % n
            return [self.early_layer_indices[idx]]

        return self.early_layer_indices

    def _exit_weights(self, active_layers: List[int]) -> dict:
        """
        Normalised weight for each active exit layer.
        w(l) = e(l) / sum_i e(l_i)
        """
        if not active_layers:
            return {}
        total = sum(self._e[l] for l in active_layers)
        if total < 1e-12:
            return {l: 1.0 / len(active_layers) for l in active_layers}
        return {l: self._e[l] / total for l in active_layers}

    # ---------------------------------------------------------------------- #
    # Random init
    # ---------------------------------------------------------------------- #

    def _random_init_weights(self):
        init_range = getattr(self.config, "initializer_range", 0.02)
        for name, p in self.named_parameters():
            n = name.lower()
            if "embed_tokens" in n:
                nn.init.normal_(p, mean=0.0, std=init_range)
            elif any(s in n for s in ("norm", "layernorm", "rmsnorm")):
                if p.dim() == 1:
                    nn.init.ones_(p)
                else:
                    nn.init.normal_(p, mean=0.0, std=init_range)
            elif p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=init_range)
            else:
                nn.init.zeros_(p)

    # ---------------------------------------------------------------------- #
    # Forward
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
        current_step=None,      # injected by LayerSkipTrainer
        **kwargs,
    ):
        if current_step is None:
            current_step = 0

        apply_layer_dropout = self.training
        compute_early_losses = (labels is not None)

        # ---- Setup -------------------------------------------------------
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

        # ---- Embedding ---------------------------------------------------
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        causal_mask = create_causal_mask(
            self.config, inputs_embeds, attention_mask,
            cache_position, past_key_values, position_ids,
        )
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # ---- Curriculum: which exit layers are active this step ----------
        if compute_early_losses:
            active_exits = self._active_exit_layers(current_step)
            weights = self._exit_weights(active_exits)
            active_exit_set = set(active_exits)
        else:
            active_exit_set = set()
            weights = {}

        # ---- Layer dropout: time-step scale S(t) -------------------------
        S_t = self._S(current_step) if apply_layer_dropout else 0.0

        # ---- Layer loop --------------------------------------------------
        early_losses = {}   # layer_idx → (weight, ce_loss_tensor)

        for i, layer in enumerate(self.model.layers):

            # --- Layer dropout (training only) ---
            if apply_layer_dropout and S_t > 0.0:
                p_drop = S_t * self._D[i] * self.p_max
                if p_drop > 0.0 and torch.rand(1, device=device).item() < p_drop:
                    # Skip this layer — hidden_states pass through unchanged.
                    # During training use_cache=False so no KV slot needed.
                    pass
                else:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            # --- Early-exit CE loss (curriculum-gated) ---
            if compute_early_losses and (i in active_exit_set):
                normed = self.model.norm(hidden_states)
                exit_logits = self.lm_head(normed)

                shift_logits = exit_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                ce = loss_fct(
                    (shift_logits / max(self.loss_temp, 1e-5)).view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                early_losses[i] = (weights[i], ce)

        # ---- Final exit --------------------------------------------------
        final_hidden = self.model.norm(hidden_states)
        logits = self.lm_head(final_hidden)

        # ---- Loss --------------------------------------------------------
        total_loss = None
        loss_dict = {}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            final_ce = loss_fct(
                (shift_logits / max(self.loss_temp, 1e-5)).view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss_dict["final_ce_loss"] = final_ce
            total_loss = final_ce

            for layer_idx, (w, ce) in early_losses.items():
                loss_dict[f"early_ce_loss_{layer_idx}"] = ce
                total_loss = total_loss + w * ce

            loss_dict["total_loss"] = total_loss
            loss_dict["n_active_exits"] = float(len(early_losses))
            loss_dict["S_t"] = S_t

        return LayerSkipOutput(
            loss=total_loss,
            logits=logits,
            past_key_values=past_key_values,
            loss_dict=loss_dict if loss_dict else None,
        )

    # ---------------------------------------------------------------------- #
    # Parameter summary
    # ---------------------------------------------------------------------- #

    def print_parameter_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        L = self.config.num_hidden_layers
        print(f"[LAYERSKIP] Total params:           {total/1e6:.2f}M")
        print(f"[LAYERSKIP] Trainable params:       {trainable/1e6:.2f}M")
        print(f"[LAYERSKIP] Extra params over base: 0  (shared lm_head, no additions)")
        print(f"[LAYERSKIP] p_max:                  {self.p_max}")
        print(f"[LAYERSKIP] dropout_schedule:       {self.dropout_schedule}")
        print(f"[LAYERSKIP] escale:                 {self.escale}")
        print(f"[LAYERSKIP] curriculum_type:        {self.curriculum_type}")
        print(f"[LAYERSKIP] exit layers ({len(self.early_layer_indices)}): {self.early_layer_indices}")
        print(f"[LAYERSKIP] D(l) ramp (exp): {[f'{d:.3f}' for d in self._D]}")
