import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache


def get_Nvium_inference_model_class(BaseModelClass):
    """
    Wraps a trained multi-exit model for fast inference with cascaded
    routing and multi-level piggybacking.

    Assumptions (for maximum speed):
      - batch_size = 1
      - No padding (no attention_mask needed)
      - Routers are always MLP (no cache)
      - Early and final adapters are always present

    Architecture (example with exits at layers 6, 12, 18 out of 24):
      - Segment 0: layers  0– 5  (always run)
      - Segment 1: layers  6–11  (skipped by exit-0 tokens)
      - Segment 2: layers 12–17  (skipped by exit-0 and exit-1 tokens)
      - Segment 3: layers 18–23  (skipped by exit-0, -1, and -2 tokens)

    Piggybacking:
      When a token continues past an exit point, any tokens previously
      deferred at that exit are batched in ("carried") through the next
      segment.  Carried tokens ride along through all subsequent segments
      until either (a) the current token also exits (carry tokens are
      re-deferred) or (b) the current token takes the full path (carry
      tokens complete all segments and their KV caches are fully filled).

    Prefill:
      Runs all layers in one pass, computes p_mix from all exits.
    """

    class InferenceMultiHeadModel(BaseModelClass):
        def __init__(self, config, exp_config=None, **kwargs):
            if exp_config is None:
                exp_config = kwargs.pop("exp_config", None)
            if exp_config is None:
                raise ValueError("exp_config must be provided")

            super().__init__(config, exp_config=exp_config, **kwargs)

            self.history = {"decisions": [], "layers_skipped": []}

            # ---------------------------------------------------------
            # Build segments from layer boundaries
            # ---------------------------------------------------------
            num_layers = len(self.model.layers)
            exit_points = self.early_layer_indices  # e.g. [6, 12, 18]

            boundaries = [0] + exit_points + [num_layers]
            self.segments = []
            for i in range(len(boundaries) - 1):
                self.segments.append(list(self.model.layers[boundaries[i]:boundaries[i + 1]]))

            self.num_segments = len(self.segments)
            self.num_early_exits = len(exit_points)

            # Layers skipped when exiting at each exit point
            self.layers_skipped_per_exit = []
            for k in range(self.num_early_exits):
                skipped = sum(len(seg) for seg in self.segments[k + 1:])
                self.layers_skipped_per_exit.append(skipped)

            # One deferred queue per exit point
            self.deferred_queues = [[] for _ in range(self.num_early_exits)]

            # Router inference config
            router_cfg = exp_config.get("inference", {})
            self.sample_router = router_cfg.get("sample_router", False)
            self.router_sample_temp = max(float(router_cfg.get("router_sample_temp", 1.0)), 1e-5)

        # =============================================================
        # HELPERS
        # =============================================================
        def _build_causal_mask(self, cache_position, kv_len, device, dtype):
            """
            Build a position-aware 4D additive causal mask for piggybacking.
              mask[i, j] = 0        if kv_pos[j] <= q_pos[i]  (attend)
                        = -inf     otherwise                 (mask out)
            Returns shape [1, 1, q_len, kv_len].
            """
            q_pos = cache_position.unsqueeze(1)                              # [q_len, 1]
            kv_pos = torch.arange(kv_len, device=device).unsqueeze(0)         # [1, kv_len]
            causal = q_pos >= kv_pos                                          # [q_len, kv_len]
            min_val = torch.finfo(dtype).min
            mask = torch.where(
                causal,
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), min_val, dtype=dtype, device=device),
            )
            return mask.unsqueeze(0).unsqueeze(0)

        def _flush_deferred(self, past_key_values, use_cache=True):
            """
            Run any tokens still sitting in the deferred queues through their
            remaining segments so their KV entries are filled. Processes queues
            in reverse (higher exits first), so when queue[k]'s tokens traverse
            segment m > k, any higher-exit tokens that share those segments
            have already been flushed and their KV is present.
            """
            if not any(len(q) > 0 for q in self.deferred_queues):
                return

            for exit_idx in range(self.num_early_exits - 1, -1, -1):
                queue = self.deferred_queues[exit_idx]
                if not queue:
                    continue

                flush_hidden = torch.cat([q["hidden_states"] for q in queue], dim=1)
                flush_pos_ids = torch.cat([q["position_ids"] for q in queue], dim=1)
                flush_cache_pos = torch.cat([q["cache_position"] for q in queue], dim=0)

                # Sort by position so the causal mask is well-formed
                sorted_idx = flush_cache_pos.argsort()
                flush_hidden = flush_hidden[:, sorted_idx, :]
                flush_pos_ids = flush_pos_ids[:, sorted_idx]
                flush_cache_pos = flush_cache_pos[sorted_idx]

                flush_pos_embeds = self.model.rotary_emb(flush_hidden, flush_pos_ids)

                # kv_len from Python ints — avoids a GPU sync per flush.
                # Each deferred entry stores its cache position as int.
                kv_len = max(q["cache_position_int"] for q in queue) + 1
                mask = self._build_causal_mask(
                    flush_cache_pos, kv_len, flush_hidden.device, flush_hidden.dtype
                )

                flush_kwargs = dict(
                    attention_mask=mask,
                    position_ids=flush_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=flush_cache_pos,
                    position_embeddings=flush_pos_embeds,
                )

                for seg_idx in range(exit_idx + 1, self.num_segments):
                    for layer in self.segments[seg_idx]:
                        flush_hidden = layer(flush_hidden, **flush_kwargs)

            self.deferred_queues = [[] for _ in range(self.num_early_exits)]

        # =============================================================
        # FORWARD
        # =============================================================
        def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, use_cache=True, **kwargs):
            seq_len = input_ids.shape[1]
            is_decoding = (seq_len == 1)

            if past_key_values is None:
                past_key_values = DynamicCache()

            inputs_embeds = self.model.embed_tokens(input_ids)
            hidden_states = inputs_embeds

            past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, device=input_ids.device
            )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

            layer_kwargs = dict(
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            # --- SEGMENT 0: always runs ---
            for layer in self.segments[0]:
                hidden_states = layer(hidden_states, **layer_kwargs)

            if not is_decoding:
                return self._prefill_forward(
                    hidden_states, position_ids, position_embeddings,
                    layer_kwargs, past_key_values, past_seen_tokens,
                )

            # For decode, cache_position is a single position; pass it as a
            # Python int so downstream code can compute kv_len without a sync.
            return self._decode_forward(
                hidden_states, position_ids, cache_position, past_seen_tokens,
                position_embeddings, past_key_values, use_cache, layer_kwargs,
            )

        # =============================================================
        # PREFILL — run all layers, then sample last-position logits from p_mix
        # =============================================================
        def _prefill_forward(self, hidden_states, position_ids, position_embeddings,
                             layer_kwargs, past_key_values, past_seen_tokens):
            """
            Prefill:
              1. Handle deferred tokens from any previous generation.
                 - If the KV cache is reused (multi-turn), flush them through
                   the remaining segments so their KV entries are complete.
                 - Otherwise (fresh generation), just drop them.
              2. Run all remaining backbone layers to fill the KV cache for
                 the new prompt, saving the last-position hidden state at
                 each exit point.
              3. For the last position, replay the cascaded router decision
                 exactly as decode does — so the first generated token is
                 drawn from p_mix, not from the final head alone.
            """
            current_seq_len = hidden_states.shape[1]
            # past_seen_tokens is the cache size BEFORE this call;
            # after segment 0 wrote, layer-0 cache size = past_seen_tokens + current_seq_len.
            # `kv_len > current_seq_len` is equivalent to `past_seen_tokens > 0`
            # (multi-turn). All Python ints — no GPU sync.
            if any(len(q) > 0 for q in self.deferred_queues):
                if past_seen_tokens > 0:
                    self._flush_deferred(past_key_values)
                else:
                    self.deferred_queues = [[] for _ in range(self.num_early_exits)]

            # Multi-turn: build explicit 4D causal mask so new tokens can
            # attend to the full prior context (SDPA's is_causal can mishandle
            # non-square q_len < kv_len on older PyTorch).
            if past_seen_tokens > 0:
                kv_len = past_seen_tokens + current_seq_len
                cache_position = layer_kwargs["cache_position"]
                prefill_mask = self._build_causal_mask(
                    cache_position, kv_len,
                    hidden_states.device, hidden_states.dtype,
                )
                layer_kwargs = {**layer_kwargs, "attention_mask": prefill_mask}

            # Hidden state at exit point 0 (last position only).
            # No clone: layers return new tensors (no in-place mutation of
            # hidden_states), so the slice is a stable view of the prior tensor.
            exit_hidden_list = [hidden_states[:, -1:, :]]

            # Run remaining segments, recording last-position hidden state at
            # each early-exit boundary.
            for exit_idx in range(self.num_early_exits):
                for layer in self.segments[exit_idx + 1]:
                    hidden_states = layer(hidden_states, **layer_kwargs)
                if exit_idx < self.num_early_exits - 1:
                    exit_hidden_list.append(hidden_states[:, -1:, :])

            # Cascaded router decision for the last position — mirrors
            # _decode_forward so the first generated token is from p_mix.
            pos_ids_last = position_ids[:, -1:]
            pos_embeds_last = self.model.rotary_emb(hidden_states[:, -1:, :], pos_ids_last)

            B = hidden_states.shape[0]
            seq_len = current_seq_len

            def _pad_last(logits_last):
                full = logits_last.new_zeros(B, seq_len, logits_last.shape[-1])
                full[:, -1:, :] = logits_last
                return full

            for k in range(self.num_early_exits):
                e_h_norm = self.early_norms[k](exit_hidden_list[k])

                router_out = self.router_layers[k](
                    e_h_norm,
                    attention_mask=None,
                    position_ids=pos_ids_last,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=pos_embeds_last,
                )[0]
                logits_2 = self.router_heads[k](
                    self.router_final_norms[k](router_out)
                ).squeeze(1)  # [1, 2]

                if self.sample_router:
                    probs = torch.softmax(logits_2 / self.router_sample_temp, dim=-1)
                    is_exit = (torch.multinomial(probs, num_samples=1).item() == 1)
                else:
                    is_exit = (logits_2[0, 1] > logits_2[0, 0]).item()

                if is_exit:
                    early_logits = self.early_lm_heads[k](
                        e_h_norm + self.early_adapters[k](e_h_norm)
                    )
                    return CausalLMOutputWithPast(
                        logits=_pad_last(early_logits), past_key_values=past_key_values
                    )

            # Full path: final head
            final_hidden = self.model.norm(hidden_states[:, -1:, :])
            final_logits = self.lm_head(
                final_hidden + self.final_adapter(final_hidden)
            )
            return CausalLMOutputWithPast(
                logits=_pad_last(final_logits), past_key_values=past_key_values
            )

        # =============================================================
        # DECODE — cascaded routing + piggybacking
        # =============================================================
        def _decode_forward(self, hidden_states, position_ids, cache_position,
                            cache_position_int, position_embeddings,
                            past_key_values, use_cache, layer_kwargs):
            """
            Decode one token with cascaded routing.

            At each exit point:
              1. Router decides: exit or continue
              2. EXIT  → defer current token (+ re-defer any carry), return early logits
              3. CONTINUE → pick up deferred tokens, run next segment with piggybacking
            """
            carry_queue = []  # tokens being carried from earlier piggybacking

            for exit_idx in range(self.num_early_exits):
                # ---- ROUTER DECISION ----
                e_h_norm = self.early_norms[exit_idx](hidden_states)

                router_out = self.router_layers[exit_idx](
                    e_h_norm,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )[0]
                logits_2 = self.router_heads[exit_idx](
                    self.router_final_norms[exit_idx](router_out)
                ).squeeze(1)  # [1, 2]

                if self.sample_router:
                    probs = torch.softmax(logits_2 / self.router_sample_temp, dim=-1)
                    is_exit = (torch.multinomial(probs, num_samples=1).item() == 1)
                else:
                    # Index 1 = exit, index 0 = continue
                    is_exit = (logits_2[0, 1] > logits_2[0, 0]).item()

                # ---- EARLY EXIT ----
                if is_exit:
                    # Re-defer carry tokens at this level (they still need
                    # this segment onwards next time someone continues past here)
                    for q in carry_queue:
                        self.deferred_queues[exit_idx].append(q)
                    carry_queue = []

                    # Defer current token. Storing cache_position_int alongside
                    # the tensor lets later steps (carry kv_len, flush kv_len)
                    # avoid GPU syncs on .item().
                    self.deferred_queues[exit_idx].append({
                        "hidden_states": hidden_states.detach(),
                        "position_ids": position_ids,
                        "cache_position": cache_position,
                        "cache_position_int": cache_position_int,
                    })

                    # Stats
                    self.history["decisions"].append(f"early_{exit_idx + 1}")
                    self.history["layers_skipped"].append(
                        self.layers_skipped_per_exit[exit_idx]
                    )

                    # Early logits
                    logits = self.early_lm_heads[exit_idx](
                        e_h_norm + self.early_adapters[exit_idx](e_h_norm)
                    )

                    return CausalLMOutputWithPast(
                        logits=logits, past_key_values=past_key_values
                    )

                # ---- CONTINUE: run next segment with piggybacking ----
                # Pick up deferred tokens at this exit level
                if self.deferred_queues[exit_idx]:
                    carry_queue.extend(self.deferred_queues[exit_idx])
                    self.deferred_queues[exit_idx] = []

                segment = self.segments[exit_idx + 1]

                if carry_queue:
                    # Batch carry + current token
                    loop_hidden = torch.cat(
                        [q["hidden_states"] for q in carry_queue] + [hidden_states],
                        dim=1,
                    )
                    loop_position_ids = torch.cat(
                        [q["position_ids"] for q in carry_queue] + [position_ids],
                        dim=1,
                    )
                    loop_cache_position = torch.cat(
                        [q["cache_position"] for q in carry_queue] + [cache_position],
                        dim=0,
                    )

                    # Sort by position. carry_queue.extend() above can mix
                    # carried tokens (from a lower exit, higher positions)
                    # with freshly picked-up deferred tokens (lower positions),
                    # breaking the sort order the is_causal mask assumes.
                    sorted_idx = loop_cache_position.argsort()
                    loop_hidden = loop_hidden[:, sorted_idx, :]
                    loop_position_ids = loop_position_ids[:, sorted_idx]
                    loop_cache_position = loop_cache_position[sorted_idx]

                    loop_pos_embeds = self.model.rotary_emb(
                        loop_hidden, loop_position_ids
                    )

                    # Explicit position-aware causal mask.  A plain is_causal
                    # mask would be wrong: tokens in the batch can sit at
                    # positions that already exist in the KV cache (from
                    # full-path tokens), so the mask must be built from
                    # actual positions rather than tensor indices.
                    # The current token has the highest position, so kv_len
                    # is known in Python — no GPU sync.
                    kv_len = cache_position_int + 1
                    piggyback_mask = self._build_causal_mask(
                        loop_cache_position, kv_len,
                        loop_hidden.device, loop_hidden.dtype,
                    )

                    loop_kwargs = dict(
                        attention_mask=piggyback_mask,
                        position_ids=loop_position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=loop_cache_position,
                        position_embeddings=loop_pos_embeds,
                    )

                    for layer in segment:
                        loop_hidden = layer(loop_hidden, **loop_kwargs)

                    # Update carry hidden states. One sync (.tolist()) then
                    # plain Python iteration — vs one .item() per token.
                    num_carry = len(carry_queue)
                    sorted_idx_list = sorted_idx.tolist()
                    for j, orig_idx in enumerate(sorted_idx_list):
                        if orig_idx < num_carry:
                            carry_queue[orig_idx]["hidden_states"] = (
                                loop_hidden[:, j:j + 1, :].detach()
                            )

                    # Current token has the highest position, so after
                    # sorting it is always at the last index.
                    hidden_states = loop_hidden[:, -1:, :]
                else:
                    # No piggybacking — run segment on current token only
                    for layer in segment:
                        hidden_states = layer(hidden_states, **layer_kwargs)

            # ---- FULL PATH: final head ----
            # Carry tokens have completed all segments — KV caches are filled
            self.history["decisions"].append("final")
            self.history["layers_skipped"].append(0)

            final_hidden = self.model.norm(hidden_states)
            logits = self.lm_head(final_hidden + self.final_adapter(final_hidden))

            return CausalLMOutputWithPast(
                logits=logits, past_key_values=past_key_values
            )

        # =============================================================
        # GENERATE WITH STATS
        # =============================================================
        @torch.no_grad()
        def generate_with_stats(self, input_ids, **kwargs):
            # Reset stats.  Do NOT clear deferred_queues here: if the caller
            # is doing multi-turn generation with a reused KV cache, we need
            # _prefill_forward to see those deferred tokens so it can flush
            # them through the remaining segments.  If the cache is fresh,
            # _prefill_forward will detect that and drop them instead.
            self.history = {"decisions": [], "layers_skipped": []}
            return self.generate(input_ids, **kwargs)

    return InferenceMultiHeadModel