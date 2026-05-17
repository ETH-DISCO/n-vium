# Self-speculative decoding for LayerSkip (Elhoushi et al., arXiv:2404.16710).
#
# Algorithm per speculative step
# --------------------------------
#   1. DRAFT  — run only layers 0..exit_layer-1, then model.norm + lm_head.
#               Generate K tokens autoregressively (cheap).
#   2. VERIFY — run the full model on the K draft tokens in ONE batched
#               forward pass (parallels K tokens simultaneously).
#   3. ACCEPT/REJECT — standard speculative-decoding criterion:
#               accept token x with prob min(1, p_full(x) / p_draft(x)).
#               If all K accepted, sample a bonus token from p_full.
#               Otherwise resample the rejected position from the residual
#               distribution (p_full - min(p_full, p_draft)).
#
# KV-cache strategy
# --------------------------------
#   A single "master" DynamicCache tracks the full model's state.
#   Draft tokens are generated using only the lower exit_layer entries of
#   this cache (upper layers simply aren't called).
#   After acceptance of M tokens the verify forward already produced the
#   correct full KV state for all K positions; we trim it to M entries.
#
# Interface (mirrors model_inference_piggyback.py)
# --------------------------------
#   cls = get_layerskip_inference_class(LayerSkipModel)
#   model = cls(config, exp_config=cfg)
#   out, history = model.generate_with_stats(input_ids, max_new_tokens=200, ...)
#
# history keys:
#   n_accepted_per_step  — list[int]: accepted tokens per speculative step
#   n_draft_per_step     — list[int]: always draft_length
#   acceptance_rate      — float:     macro average
#   decisions            — list[str]: "spec" for each step (compat)
#   layers_skipped       — list[int]: layers saved per step

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# KV cache helpers — use DynamicCache.update() + __getitem__ for compat
# across transformers versions (some lack .key_cache / .value_cache attrs).
# ---------------------------------------------------------------------------

def _kv_layer(kv: DynamicCache, i: int):
    """Return (key, value) tensors for layer i, works across API versions."""
    # transformers >=5.x: kv.layers[i].keys / .values
    if hasattr(kv, 'layers') and hasattr(kv.layers[0], 'keys'):
        return kv.layers[i].keys, kv.layers[i].values
    # transformers <5.x: kv.key_cache[i] / kv.value_cache[i]
    return kv.key_cache[i], kv.value_cache[i]


def _kv_num_layers(kv: DynamicCache) -> int:
    if hasattr(kv, 'layers'):
        return len(kv.layers)
    return len(kv.key_cache)


def _kv_get_seen(kv: DynamicCache) -> int:
    return kv.get_seq_length()


def _build_kv(layers_kv: list, seen_tokens: int) -> DynamicCache:
    """Build a DynamicCache from a list of (key, value) tensor pairs.

    Note: `update()` concatenates to existing entries, so this must be called
    on a fresh DynamicCache. The seen_tokens counter is set by the first
    layer's update() call automatically (based on key_states.shape[-2]).
    We then correct it to `seen_tokens` for cases like trim where the
    actual sequence length differs from what the first update would infer.
    """
    new_kv = DynamicCache()
    for i, (k, v) in enumerate(layers_kv):
        new_kv.update(k, v, layer_idx=i)
    # Correct seen_tokens (update() counts from the tensor shapes,
    # but for trim/shallow_draft the intended count may differ)
    if hasattr(new_kv, '_seen_tokens'):
        new_kv._seen_tokens = seen_tokens
    return new_kv


def _clone_kv(kv: DynamicCache) -> DynamicCache:
    """Deep-copy a DynamicCache (each key/value tensor is cloned)."""
    n = _kv_num_layers(kv)
    pairs = []
    for i in range(n):
        k, v = _kv_layer(kv, i)
        pairs.append((k.clone(), v.clone()))
    return _build_kv(pairs, _kv_get_seen(kv))


def _trim_kv(kv: DynamicCache, keep_tokens: int) -> DynamicCache:
    """Slice each layer's KV to keep only the first `keep_tokens` positions."""
    n = _kv_num_layers(kv)
    pairs = []
    for i in range(n):
        k, v = _kv_layer(kv, i)
        pairs.append((k[:, :, :keep_tokens, :].clone(), v[:, :, :keep_tokens, :].clone()))
    return _build_kv(pairs, keep_tokens)


def _shallow_draft_kv(kv: DynamicCache, exit_layer: int) -> DynamicCache:
    """
    Build a draft KV cache: layers 0..exit_layer-1 are referenced,
    upper layers get zero-length placeholders.
    """
    n = _kv_num_layers(kv)
    pairs = []
    for i in range(n):
        k, v = _kv_layer(kv, i)
        if i < exit_layer:
            pairs.append((k, v))
        else:
            pairs.append((k[:, :, :0, :], v[:, :, :0, :]))
    return _build_kv(pairs, _kv_get_seen(kv))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_layerskip_inference_class(BaseModelClass):
    """
    Wraps a LayerSkipModel (or any LlamaForCausalLM subclass) with
    self-speculative decoding.  The returned class has the same constructor
    signature as the base class.
    """

    class LayerSkipInferenceModel(BaseModelClass):

        def __init__(self, config, exp_config=None, **kwargs):
            if exp_config is None:
                exp_config = kwargs.pop("exp_config", None)
            if exp_config is None:
                raise ValueError("exp_config must be provided")
            super().__init__(config, exp_config=exp_config, **kwargs)
            self.history = {
                "n_accepted_per_step": [],
                "n_draft_per_step": [],
                "decisions": [],
                "layers_skipped": [],
            }
            # Default speculative decoding config — set these before calling generate().
            # exit_layer=None means L//3 is used at runtime.
            ls_cfg = exp_config.get("layerskip", {})
            L = config.num_hidden_layers
            self.spec_exit_layer = int(ls_cfg.get("exit_layer", L // 3))
            self.spec_draft_length = int(ls_cfg.get("draft_length", 5))

        # ------------------------------------------------------------------ #
        # Draft forward — runs only layers 0..exit_layer-1
        # ------------------------------------------------------------------ #

        def _forward_draft(
            self,
            input_ids: torch.Tensor,
            past_key_values: DynamicCache,
            exit_layer: int,
            position_ids=None,
            attention_mask=None,
        ) -> CausalLMOutputWithPast:
            """
            One-token-at-a-time forward through only the first `exit_layer`
            transformer layers, then model.norm + lm_head.
            `past_key_values` must already contain entries for all layers
            (upper layers will have zero-length KV but that's fine — they are
            not called).
            """
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

            past_seen = past_key_values.get_seq_length()
            cache_position = torch.arange(past_seen, past_seen + seq_len, device=device)

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

            hidden_states = self.model.embed_tokens(input_ids)

            causal_mask = create_causal_mask(
                self.config, hidden_states, attention_mask,
                cache_position, past_key_values, position_ids,
            )
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

            for i in range(exit_layer):
                hidden_states = self.model.layers[i](
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            normed = self.model.norm(hidden_states)
            logits = self.lm_head(normed)

            return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

        # ------------------------------------------------------------------ #
        # Full forward — all layers
        # ------------------------------------------------------------------ #

        def _forward_full(
            self,
            input_ids: torch.Tensor,
            past_key_values: DynamicCache,
            position_ids=None,
            attention_mask=None,
        ) -> CausalLMOutputWithPast:
            """
            Full-model forward (all L layers) in one call.  Supports batched
            input (seq_len > 1) for the verification step.
            """
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

            past_seen = past_key_values.get_seq_length()
            cache_position = torch.arange(past_seen, past_seen + seq_len, device=device)

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

            hidden_states = self.model.embed_tokens(input_ids)

            causal_mask = create_causal_mask(
                self.config, hidden_states, attention_mask,
                cache_position, past_key_values, position_ids,
            )
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

            for layer in self.model.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            normed = self.model.norm(hidden_states)
            logits = self.lm_head(normed)

            return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

        # ------------------------------------------------------------------ #
        # One speculative decoding step
        # ------------------------------------------------------------------ #

        @torch.no_grad()
        def _speculative_step(
            self,
            master_kv: DynamicCache,
            last_token_id: int,
            exit_layer: int,
            draft_length: int,
            temperature: float,
            do_sample: bool,
            device: torch.device,
        ):
            """
            Executes one speculative decoding step:
              - Draft `draft_length` tokens via the shallow model.
              - Verify them with the full model (single batched pass).
              - Accept / reject and return (accepted_tokens, n_accepted, updated_kv).

            Returns
            -------
            accepted_tokens : list[int]
            n_accepted       : int  (number of original draft tokens accepted,
                                     bonus token not counted)
            new_master_kv    : DynamicCache  (updated with accepted tokens)
            """
            prompt_len = master_kv.get_seq_length()

            # ----------------------------------------------------------
            # DRAFT PHASE
            # ----------------------------------------------------------
            # Build a draft KV that shares the lower-layer tensors with master_kv.
            # Upper layers are zero-length placeholders — never called.
            draft_kv = _shallow_draft_kv(master_kv, exit_layer)
            # Detach references so draft appends don't corrupt master_kv upper layers.
            # (Lower layers ARE shared; draft calls will extend them — we correct below.)
            # Actually we need to clone the lower layers too so draft doesn't corrupt master.
            n_layers = _kv_num_layers(master_kv)
            pairs = []
            for i in range(n_layers):
                k, v = _kv_layer(master_kv, i)
                if i < exit_layer:
                    pairs.append((k.clone(), v.clone()))
                else:
                    pairs.append((k[:, :, :0, :], v[:, :, :0, :]))
            draft_kv_clean = _build_kv(pairs, _kv_get_seen(master_kv))

            draft_tokens: list[int] = []
            draft_probs: list[torch.Tensor] = []   # each: [vocab_size]

            current_id = last_token_id
            for _ in range(draft_length):
                tok_tensor = torch.tensor([[current_id]], device=device)
                out = self._forward_draft(tok_tensor, draft_kv_clean, exit_layer)
                logit = out.logits[:, -1, :]   # [1, V]
                prob = _logit_to_prob(logit, temperature)  # [V]

                if do_sample:
                    tok_id = torch.multinomial(prob, 1).item()
                else:
                    tok_id = prob.argmax().item()

                draft_tokens.append(tok_id)
                draft_probs.append(prob)
                current_id = tok_id

            # ----------------------------------------------------------
            # VERIFY PHASE — one batched forward through ALL layers
            # ----------------------------------------------------------
            # Clone master_kv so verify doesn't mutate it until we know
            # which tokens are accepted.
            verify_kv = _clone_kv(master_kv)

            # Include last_token_id so verify logits align with draft positions:
            #   logits[:, 0, :] = p_full(· | prefix, last_token_id)  → verifies d0
            #   logits[:, i, :] = p_full(· | prefix, last_token_id, d0..d_{i-1}) → verifies d_i
            #   logits[:, K, :] = p_full(· | prefix, last_token_id, d0..d_{K-1}) → bonus token
            verify_input = torch.tensor(
                [[last_token_id] + draft_tokens], device=device
            )  # [1, K+1]
            verify_out = self._forward_full(verify_input, verify_kv)
            full_logits = verify_out.logits  # [1, K+1, V]

            # verify_kv now has prompt_len + K + 1 tokens for all layers.

            # ----------------------------------------------------------
            # ACCEPT / REJECT
            # ----------------------------------------------------------
            accepted_tokens: list[int] = []
            n_accepted = 0

            for i in range(draft_length):
                full_prob = _logit_to_prob(full_logits[:, i, :], temperature)  # [V]
                q = draft_probs[i]     # [V]
                p = full_prob          # [V]
                tok = draft_tokens[i]

                accept_prob = min(1.0, (p[tok] / (q[tok] + 1e-9)).item())

                if torch.rand(1).item() < accept_prob:
                    accepted_tokens.append(tok)
                    n_accepted += 1
                else:
                    # Sample from residual distribution
                    corrected = (p - torch.min(p, q)).clamp(min=0.0)
                    z = corrected.sum()
                    if z > 1e-8:
                        corrected = corrected / z
                        new_tok = torch.multinomial(corrected, 1).item()
                    else:
                        new_tok = p.argmax().item()
                    accepted_tokens.append(new_tok)
                    # Trim verify_kv: keep prompt + last_token_id + n_accepted draft tokens.
                    # The resampled token's KV is not in verify_kv — next step recomputes.
                    new_master_kv = _trim_kv(verify_kv, prompt_len + 1 + n_accepted)
                    return accepted_tokens, n_accepted, new_master_kv

            # All K tokens accepted — sample a bonus token from full_logits[:, K, :]
            # (the logit after seeing last_token_id + all K draft tokens).
            bonus_prob = _logit_to_prob(full_logits[:, draft_length, :], temperature)
            if do_sample:
                bonus_tok = torch.multinomial(bonus_prob, 1).item()
            else:
                bonus_tok = bonus_prob.argmax().item()
            accepted_tokens.append(bonus_tok)

            # master_kv = verify_kv trimmed to prompt + last_token_id + K draft tokens
            new_master_kv = _trim_kv(verify_kv, prompt_len + 1 + draft_length)
            return accepted_tokens, n_accepted, new_master_kv

        # ------------------------------------------------------------------ #
        # Main generation entry point
        # ------------------------------------------------------------------ #

        @torch.no_grad()
        def generate_with_stats(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 200,
            exit_layer: int = None,
            draft_length: int = 5,
            temperature: float = 1.0,
            do_sample: bool = False,
            **kwargs,
        ):
            """
            Generate `max_new_tokens` tokens using self-speculative decoding.

            Parameters
            ----------
            input_ids     : [1, T]  prompt token ids
            max_new_tokens: int
            exit_layer    : int  — which layer to exit for drafting.
                                   Defaults to num_hidden_layers // 2.
            draft_length  : int  — K, tokens drafted per step
            temperature   : float
            do_sample     : bool

            Returns
            -------
            generated_ids : [1, T + n_generated]
            history       : dict  with stats
            """
            self.history = {
                "n_accepted_per_step": [],
                "n_draft_per_step": [],
                "decisions": [],
                "layers_skipped": [],
            }

            device = input_ids.device
            L = len(self.model.layers)

            if exit_layer is None:
                exit_layer = L // 2

            # Clamp to valid range
            exit_layer = max(1, min(exit_layer, L - 1))
            layers_saved_per_accepted = L - exit_layer

            generated = input_ids.clone()   # [1, T]

            # Encode prompt with full model to warm up the KV cache
            master_kv = DynamicCache()
            prompt_out = self._forward_full(input_ids, master_kv)
            # master_kv now has T entries for all L layers.

            # Last token predicted by the full model on the prompt
            last_logit = prompt_out.logits[:, -1, :]
            last_prob = _logit_to_prob(last_logit, temperature)
            if do_sample:
                last_tok = torch.multinomial(last_prob, 1).item()
            else:
                last_tok = last_prob.argmax().item()
            generated = torch.cat([generated, torch.tensor([[last_tok]], device=device)], dim=1)
            n_generated = 1

            while n_generated < max_new_tokens:
                remaining = max_new_tokens - n_generated
                K = min(draft_length, remaining)

                accepted, n_acc, master_kv = self._speculative_step(
                    master_kv=master_kv,
                    last_token_id=generated[0, -1].item(),
                    exit_layer=exit_layer,
                    draft_length=K,
                    temperature=temperature,
                    do_sample=do_sample,
                    device=device,
                )

                # Append accepted tokens
                acc_tensor = torch.tensor([accepted], device=device)
                generated = torch.cat([generated, acc_tensor], dim=1)
                n_generated += len(accepted)

                # Tracking
                self.history["n_accepted_per_step"].append(n_acc)
                self.history["n_draft_per_step"].append(K)
                self.history["decisions"].append("spec")
                self.history["layers_skipped"].append(n_acc * layers_saved_per_accepted)

                # EOS check
                if accepted[-1] == self.config.eos_token_id:
                    break

            # Aggregate stats
            total_draft = sum(self.history["n_draft_per_step"])
            total_accepted = sum(self.history["n_accepted_per_step"])
            self.history["acceptance_rate"] = (
                total_accepted / total_draft if total_draft > 0 else 0.0
            )
            self.history["mean_accepted_per_step"] = (
                total_accepted / len(self.history["n_accepted_per_step"])
                if self.history["n_accepted_per_step"] else 0.0
            )

            return generated, self.history

        def generate(self, input_ids=None, attention_mask=None,
                     max_new_tokens=200, min_new_tokens=None,
                     do_sample=False, temperature=1.0,
                     pad_token_id=None, eos_token_id=None, **kwargs):
            """
            Drop-in replacement for HF generate() — routes to self-speculative
            decoding using self.spec_exit_layer and self.spec_draft_length.

            Accepts the same kwargs as measure.py passes to model.generate():
              min_new_tokens, max_new_tokens, do_sample, pad_token_id, eos_token_id
            Ignores eos_token_id=-1 (strict-length mode from the benchmark).
            """
            # measure.py passes eos_token_id=-1 to force exact length; honour max_new_tokens.
            n_tokens = max_new_tokens or 200

            generated, _ = self.generate_with_stats(
                input_ids,
                max_new_tokens=n_tokens,
                exit_layer=self.spec_exit_layer,
                draft_length=self.spec_draft_length,
                temperature=temperature if temperature else 1.0,
                do_sample=do_sample,
            )
            return generated

        # compat alias
        def generate_speculative(self, *args, **kwargs):
            return self.generate_with_stats(*args, **kwargs)

    return LayerSkipInferenceModel


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _logit_to_prob(logit: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    logit : [1, V] or [V]   →   prob : [V]  (sum-to-1, on same device)
    """
    logit = logit.squeeze(0).float()
    if temperature < 1e-5:
        # Greedy: one-hot distribution
        prob = torch.zeros_like(logit)
        prob[logit.argmax()] = 1.0
        return prob
    return F.softmax(logit / temperature, dim=-1)
