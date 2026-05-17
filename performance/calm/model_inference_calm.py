# model_inference_calm.py
#
# Per-token adaptive early-exit decoding for CALM (Schuster et al., 2022).
#
# Algorithm per generated token
# --------------------------------
#   1. Run layers 0, 1, 2, ... sequentially on the single new token.
#   2. At each exit layer l in exit_set (and at the final layer), compute
#        logits_l = lm_head(exit_norm_l(hidden_l))   (or model.norm for final)
#        conf_l  = max(softmax(logits_l))
#      If conf_l > threshold (or l is final), emit argmax(logits_l) and STOP.
#   3. STATE PROPAGATION (CALM paper, App. C):
#      For every skipped upper layer j > exit_layer, populate its KV cache
#      entry for the current token by running only its k_proj/v_proj on the
#      exit-layer hidden state (post input_layernorm), with rotary applied.
#      Attention and FFN are skipped → real compute saving.
#
# KV cache
# --------------------------------
#   A single master DynamicCache tracks all L layers. Prompt is processed
#   with a full-model forward (warm-up). During decoding, lower layers
#   extend their cache via normal layer.forward(); upper layers get filled
#   via manual past_key_values.update() (state propagation).
#
# Dense baseline bypass
# --------------------------------
#   The class inherits from LlamaForCausalLM → LlamaForCausalLM.generate()
#   can be called directly on instances via MRO, bypassing the override.

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def get_calm_inference_class(BaseModelClass):
    """
    Wrap CalmModel (or any LlamaForCausalLM subclass that defines
    self.exit_layer_indices / self.exit_norms / self.per_exit_norms_enabled)
    with adaptive early-exit generation.
    """

    class CalmInferenceModel(BaseModelClass):

        def __init__(self, config, exp_config=None, **kwargs):
            if exp_config is None:
                exp_config = kwargs.pop("exp_config", None)
            if exp_config is None:
                raise ValueError("exp_config must be provided")
            super().__init__(config, exp_config=exp_config, **kwargs)

            calm_cfg = exp_config.get("calm", {})
            self.spec_threshold = float(calm_cfg.get("threshold", 0.9))

            self.history = {
                "exits": [],
                "n_tokens_generated": 0,
                "n_early_exits": 0,
                "avg_exit_layer": 0.0,
                "early_exit_frac": 0.0,
            }

        # ------------------------------------------------------------------ #
        # Full forward — used for prompt warm-up and dense verification
        # ------------------------------------------------------------------ #

        def _forward_full(self, input_ids, past_key_values):
            B, S = input_ids.shape
            device = input_ids.device

            past_seen = past_key_values.get_seq_length()
            cache_position = torch.arange(past_seen, past_seen + S, device=device)
            position_ids = cache_position.unsqueeze(0).expand(B, -1)

            hidden = self.model.embed_tokens(input_ids)
            causal_mask = create_causal_mask(
                self.config, hidden, None,
                cache_position, past_key_values, position_ids,
            )
            position_embeddings = self.model.rotary_emb(hidden, position_ids)

            for layer in self.model.layers:
                hidden = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            normed = self.model.norm(hidden)
            logits = self.lm_head(normed)
            return logits, past_key_values

        # ------------------------------------------------------------------ #
        # Adaptive early-exit step for a single new token
        # ------------------------------------------------------------------ #

        @staticmethod
        def _sample_token(logits, do_sample, temperature):
            """Greedy or multinomial sample from [1, V] logits. Returns int."""
            if not do_sample or temperature <= 0:
                return int(logits.argmax(dim=-1).item())
            scaled = logits.float() / max(temperature, 1e-5)
            probs = F.softmax(scaled, dim=-1).squeeze(0)
            return int(torch.multinomial(probs, num_samples=1).item())

        def _forward_step_adaptive(self, input_ids, past_key_values, threshold,
                                    do_sample=False, temperature=1.0):
            """
            Runs one decoding step on input_ids = [1, 1]. Lower layers extend
            their KV cache normally; once confidence > threshold (or final
            layer reached), state-propagates the exit hidden state into all
            remaining upper layers' KV caches. Returns (token_id, exit_layer).
            """
            B, S = input_ids.shape
            device = input_ids.device

            past_seen = past_key_values.get_seq_length()
            cache_position = torch.arange(past_seen, past_seen + S, device=device)
            position_ids = cache_position.unsqueeze(0).expand(B, -1)

            hidden = self.model.embed_tokens(input_ids)
            causal_mask = create_causal_mask(
                self.config, hidden, None,
                cache_position, past_key_values, position_ids,
            )
            position_embeddings = self.model.rotary_emb(hidden, position_ids)
            cos, sin = position_embeddings

            L = self.config.num_hidden_layers
            final_idx = L - 1
            exit_at = final_idx
            next_token = None

            for i, layer in enumerate(self.model.layers):
                hidden = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                is_exit_point = (i in self._exit_set) or (i == final_idx)
                if not is_exit_point:
                    continue

                if i == final_idx:
                    normed = self.model.norm(hidden)
                elif self.per_exit_norms_enabled:
                    normed = self.exit_norms[str(i)](hidden)
                else:
                    normed = self.model.norm(hidden)

                logits = self.lm_head(normed)  # [B, S, V]
                last_logit = logits[:, -1, :]
                probs = F.softmax(last_logit.float(), dim=-1)
                # CALM "softmax response": top-1 minus top-2 softmax value.
                top2 = probs.topk(2, dim=-1).values    # [B, 2]
                conf = (top2[:, 0] - top2[:, 1]).item()

                if (i == final_idx) or (conf > threshold):
                    next_token = self._sample_token(last_logit, do_sample, temperature)
                    exit_at = i
                    break

            # --- State propagation for skipped upper layers ---
            if exit_at < final_idx:
                num_kv_heads = self.config.num_key_value_heads
                head_dim = getattr(
                    self.config, "head_dim",
                    self.config.hidden_size // self.config.num_attention_heads,
                )
                for j in range(exit_at + 1, L):
                    upper = self.model.layers[j]
                    h_in = upper.input_layernorm(hidden)                       # [B, S, H]
                    k = upper.self_attn.k_proj(h_in)                           # [B, S, nkv*hd]
                    v = upper.self_attn.v_proj(h_in)
                    k = k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)   # [B, nkv, S, hd]
                    v = v.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
                    # rotary on k (dummy q to satisfy signature)
                    dummy_q = torch.empty_like(k)
                    _, k_rot = apply_rotary_pos_emb(dummy_q, k, cos, sin)
                    past_key_values.update(k_rot, v, layer_idx=j)

            return next_token, exit_at

        # ------------------------------------------------------------------ #
        # Main generation entry point
        # ------------------------------------------------------------------ #

        @torch.no_grad()
        def generate_with_stats(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 128,
            threshold: float = None,
            do_sample: bool = False,
            temperature: float = 1.0,
            eos_token_id: int = None,
            **kwargs,
        ):
            """
            Generate max_new_tokens tokens using CALM adaptive early exit.
            Greedy decoding (do_sample=False is the only supported mode here —
            speedup measurement doesn't need sampling).

            If eos_token_id is None, stops at config.eos_token_id. Set
            eos_token_id=-1 to disable early stopping and force exact length.
            """
            self.history = {
                "exits": [],
                "n_tokens_generated": 0,
                "n_early_exits": 0,
                "avg_exit_layer": 0.0,
                "early_exit_frac": 0.0,
            }

            if threshold is None:
                threshold = self.spec_threshold

            L = self.config.num_hidden_layers
            final_idx = L - 1
            device = input_ids.device

            if eos_token_id is None:
                eos_token_id = self.config.eos_token_id

            # --- Prompt warm-up: full forward to populate KV for all layers ---
            master_kv = DynamicCache()
            prompt_logits, master_kv = self._forward_full(input_ids, master_kv)
            first_tok = self._sample_token(
                prompt_logits[:, -1, :], do_sample, temperature
            )
            generated = torch.cat(
                [input_ids, torch.tensor([[first_tok]], device=device)], dim=1
            )
            n_generated = 1
            # Prompt's own prediction uses the full model → count as final-layer exit.
            self.history["exits"].append(final_idx)

            while n_generated < max_new_tokens:
                current = generated[:, -1:].contiguous()
                next_tok, exit_at = self._forward_step_adaptive(
                    current, master_kv, threshold,
                    do_sample=do_sample, temperature=temperature,
                )
                generated = torch.cat(
                    [generated, torch.tensor([[next_tok]], device=device)], dim=1
                )
                n_generated += 1
                self.history["exits"].append(exit_at)
                if exit_at < final_idx:
                    self.history["n_early_exits"] += 1
                if eos_token_id is not None and eos_token_id != -1 and next_tok == eos_token_id:
                    break

            self.history["n_tokens_generated"] = n_generated
            n_ex = len(self.history["exits"])
            self.history["avg_exit_layer"] = sum(self.history["exits"]) / max(n_ex, 1)
            self.history["early_exit_frac"] = self.history["n_early_exits"] / max(n_ex, 1)
            return generated, self.history

        @torch.no_grad()
        def generate_dense(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 128,
            do_sample: bool = False,
            temperature: float = 1.0,
            eos_token_id: int = -1,
            **kwargs,
        ):
            """
            Bare-loop dense generation — same control flow as generate_with_stats
            but skips the early-exit check entirely (always runs to final layer).
            This is the apples-to-apples dense baseline for measuring adaptive
            speedup: both modes share the exact same prompt/token/cache machinery
            and only differ in the per-token forward (full vs early-exit).
            """
            device = input_ids.device

            master_kv = DynamicCache()
            prompt_logits, master_kv = self._forward_full(input_ids, master_kv)
            first_tok = self._sample_token(
                prompt_logits[:, -1, :], do_sample, temperature
            )
            generated = torch.cat(
                [input_ids, torch.tensor([[first_tok]], device=device)], dim=1
            )
            n_generated = 1

            while n_generated < max_new_tokens:
                current = generated[:, -1:].contiguous()
                logits, master_kv = self._forward_full(current, master_kv)
                next_tok = self._sample_token(
                    logits[:, -1, :], do_sample, temperature
                )
                generated = torch.cat(
                    [generated, torch.tensor([[next_tok]], device=device)], dim=1
                )
                n_generated += 1
                if eos_token_id is not None and eos_token_id != -1 and next_tok == eos_token_id:
                    break

            return generated

        def generate(self, input_ids=None, attention_mask=None,
                     max_new_tokens=200, min_new_tokens=None,
                     do_sample=False, temperature=1.0,
                     pad_token_id=None, eos_token_id=None, **kwargs):
            """
            Drop-in replacement for HF generate() — routes to adaptive-exit
            decoding using self.spec_threshold. Matches the layerskip
            inference wrapper's interface so benchmarks can call
            model.generate() interchangeably.
            """
            n_tokens = max_new_tokens or 200
            gen, _ = self.generate_with_stats(
                input_ids,
                max_new_tokens=n_tokens,
                threshold=self.spec_threshold,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
            )
            return gen

    return CalmInferenceModel
