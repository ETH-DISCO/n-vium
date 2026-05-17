import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import ModelOutput
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from typing import Optional, Tuple, List
from dataclasses import dataclass
from Nvium_utils import freeze_model_parameters, print_run, compute_distribution_metrics, create_router
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

# -----------------------------------------------------------------------------
# Global Settings
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------------------
# Helper Modules
# -----------------------------------------------------------------------------
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    sum_val = (x * mask).sum()
    count = mask.sum()
    return sum_val / count.clamp(min=1.0)


@dataclass
class CustomModelOutput(CausalLMOutputWithPast):
    loss_dict: Optional[dict] = None
    router_weights: Optional[List[torch.FloatTensor]] = None
    exit_probabilities: Optional[torch.FloatTensor] = None


class NviumModel(LlamaForCausalLM):
    def __init__(self, config, exp_config=None, **kwargs):
        attn_impl = kwargs.pop("attn_implementation", None)
        super().__init__(config, **kwargs)
        if attn_impl is not None:
            self.config._attn_implementation = attn_impl
        
        if exp_config is None:
            raise ValueError("exp_config must be provided")
        
        self.cfg = exp_config
        
        # CRITICAL: Support multiple early exit layers
        early_cfg = self.cfg.get("early_layer_idx")
        if isinstance(early_cfg, list):
            self.early_layer_indices = sorted(early_cfg)
        else:
            self.early_layer_indices = [int(early_cfg)]
        
        self.num_exits = len(self.early_layer_indices) + 1  # +1 for final
        
        # Router Config
        router_cfg = self.cfg.get("router", {})
        self.router_temp = float(router_cfg.get("router_temp", 1.0))
        
        # Loss & Alphas
        loss_cfg = self.cfg.get("loss", {})
        self.warmup_steps_limit = int(loss_cfg.get("warmup_steps", 0))
        self.loss_temp = float(self.cfg.get("temperature", 1.0))
        self.alpha_map_warmup = self._parse_alphas(loss_cfg, "warmup")
        self.alpha_map_main = self._parse_alphas(loss_cfg, "main")
        
        # CREATE MULTIPLE EARLY EXITS
        self.early_norms = nn.ModuleList()
        self.early_lm_heads = nn.ModuleList()
        self.early_adapters = nn.ModuleList()
        self.router_layers = nn.ModuleList()
        self.router_final_norms = nn.ModuleList()
        self.router_heads = nn.ModuleList()
        
        # =====================================================================
        # EARLY ADAPTERS — one per exit point
        # =====================================================================
        adapter_cfg = self.cfg.get("early_adapter", {})
        self.use_early_adapter = adapter_cfg.get("enabled", False)
        
        for i, layer_idx in enumerate(self.early_layer_indices):
            # Norm
            self.early_norms.append(copy.deepcopy(self.model.norm))
                        
            # LM Head
            early_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.early_lm_heads.append(early_head)
            
            # Early Adapter
            if self.use_early_adapter:
                adapter_dim = int(adapter_cfg.get("dim", config.hidden_size))
                adapter = nn.Sequential(
                    nn.Linear(config.hidden_size, adapter_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(adapter_dim, config.hidden_size, bias=False),
                )
                self.early_adapters.append(adapter)
                adapter_params = sum(p.numel() for p in adapter.parameters())
                print_run("INIT", f"Early adapter {i} at layer {layer_idx}: {config.hidden_size} -> {adapter_dim} -> {config.hidden_size} ({adapter_params / 1e6:.2f}M params)")
            
            # Router
            router_layer, router_output_dim = create_router(config, router_cfg)
            self.router_layers.append(router_layer)
            
            # Router Final Norm
            router_norm = LlamaRMSNorm(router_output_dim, eps=config.rms_norm_eps)
            self.router_final_norms.append(router_norm)
            
            # Router Head (binary: continue or exit)
            router_head = nn.Linear(router_output_dim, 2)
            self.router_heads.append(router_head)

        # =====================================================================
        # FINAL ADAPTER
        # =====================================================================
        final_adapter_cfg = self.cfg.get("final_adapter", {})
        self.use_final_adapter = final_adapter_cfg.get("enabled", False)
        if self.use_final_adapter:
            final_adapter_dim = int(final_adapter_cfg.get("dim", config.hidden_size))
            self.final_adapter = nn.Sequential(
                nn.Linear(config.hidden_size, final_adapter_dim, bias=False),
                nn.SiLU(),
                nn.Linear(final_adapter_dim, config.hidden_size, bias=False),
            )
            fa_params = sum(p.numel() for p in self.final_adapter.parameters())
            print_run("INIT", f"Final adapter: {config.hidden_size} -> {final_adapter_dim} -> {config.hidden_size} ({fa_params / 1e6:.2f}M params)")

        self.decode_head = self.cfg.get("decode_head", "final")

        if self.cfg.get("random_init_all", True):
            self._random_init_all_parameters_like_pretraining()

        self.reset_all_router_parameters()
        self._apply_tying(self.cfg.get("tying", {}))
        
        if "freeze_options" in self.cfg:
            freeze_model_parameters(self, self.cfg)
        
        self.active_warmup = {k: (v > 0.0) for k, v in self.alpha_map_warmup.items()}
        self.active_main = {k: (v > 0.0) for k, v in self.alpha_map_main.items()}

        grad_accum = self.cfg.get("training_args", {}).get("gradient_accumulation_steps", 1)
        logging_steps = self.cfg.get("training_args", {}).get("logging_steps", 5)
        self.light_interval = logging_steps * grad_accum
        self.heavy_interval = 25 * grad_accum

        self._print_component_param_counts()

    def _parse_alphas(self, loss_cfg, phase_name):
        phase_cfg = loss_cfg.get(phase_name, {})
        alpha_vals = loss_cfg.get(f"alphas_{phase_name}", {})
        terms = ["mix_loss", "router_warmup_loss", "router_compute_loss"]
        result = {}
        for t in terms:
            is_active = phase_cfg.get(t, False)
            weight = float(alpha_vals.get(t, 1.0))
            result[t] = weight if is_active else 0.0
        return result

    def _apply_tying(self, tie_cfg: dict):
        tie_final = tie_cfg.get("tie_final_to_embeddings", True)
        tie_early_to_final = tie_cfg.get("tie_early_to_final", False)
        tie_early_to_emb = tie_cfg.get("tie_early_to_embeddings", False)
        init_early_from = tie_cfg.get("init_early_from", "final")

        if tie_final:
            self.lm_head.weight = self.model.embed_tokens.weight
            self.config.tie_word_embeddings = True
        else:
            if self.lm_head.weight is self.model.embed_tokens.weight:
                self.lm_head.weight = nn.Parameter(self.lm_head.weight.detach().clone())
            self.config.tie_word_embeddings = False

        # Apply tying to ALL early heads
        tied_keys = []
        for i, early_head in enumerate(self.early_lm_heads):
            if tie_early_to_final:
                early_head.weight = self.lm_head.weight
                tied_keys.append(f"early_lm_heads.{i}.weight")
            elif tie_early_to_emb:
                early_head.weight = self.model.embed_tokens.weight
                tied_keys.append(f"early_lm_heads.{i}.weight")
            else:
                if init_early_from == "final":
                    early_head.weight.data.copy_(self.lm_head.weight.data)
                elif init_early_from == "embeddings":
                    early_head.weight.data.copy_(self.model.embed_tokens.weight.data)

        self._early_lm_head_tied_keys = tied_keys

    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = self.state_dict()
        # Remove tied early head weights — they share storage with lm_head.weight
        # and will be re-tied on load via _apply_tying, matching single-head behaviour
        for k in self._early_lm_head_tied_keys:
            state_dict.pop(k, None)
        super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)

    def reset_all_router_parameters(self):
        """Reset all routers matching single-exit init strategy."""
        print_run("INIT", f"Resetting {len(self.router_layers)} router(s) parameters...")

        for router_idx, (router_layer, router_head) in enumerate(zip(self.router_layers, self.router_heads)):
            print_run("INIT", f"  Router {router_idx} at layer {self.early_layer_indices[router_idx]}...")

            # 1. Standard Init for router body
            router_layer.apply(self._init_weights)

            # 2. Router layer is not zero-initialized on purpose
            if hasattr(router_layer, "layers"):
                last_layer = router_layer.layers[-1]
                if hasattr(last_layer, "fc2"):
                    if last_layer.fc2.bias is not None:
                        nn.init.zeros_(last_layer.fc2.bias)

            # 3. Scale-invariant router_head init
            #    Var(logit) ≈ fan_in × std² → target logit_std ≈ 0.4
            #    so softmax outputs ≈ [0.44, 0.56] at init regardless of hidden_dim
            fan_in = router_head.in_features
            target_logit_std = float(self.cfg.get("router", {}).get("router_head_init_std", 0.4))
            nn.init.normal_(router_head.weight, mean=0.0, std=target_logit_std / math.sqrt(fan_in))
            if router_head.bias is not None:
                nn.init.zeros_(router_head.bias)

            print_run("INIT", f"  Router {router_idx} head init: fan_in={fan_in}, weight_std={target_logit_std / math.sqrt(fan_in):.6f}")

        print_run("INIT", "All routers initialized (zero-init MLP output + scaled head).")

    def _print_component_param_counts(self):
        def count(module):
            return sum(p.numel() for p in module.parameters())
        
        total_unique = sum(p.numel() for p in set(self.parameters()))
        base = count(self.model) + count(self.lm_head)
        
        router_params = sum(
            count(rl) + count(rn) + count(rh) 
            for rl, rn, rh in zip(self.router_layers, self.router_final_norms, self.router_heads)
        )
        
        early_norm_params = sum(count(n) for n in self.early_norms)
        
        # Only count early_lm_heads if they have unique weights
        early_head_params = early_norm_params
        for early_head in self.early_lm_heads:
            early_head_is_tied = (early_head.weight is self.lm_head.weight or 
                                  early_head.weight is self.model.embed_tokens.weight)
            if not early_head_is_tied:
                early_head_params += count(early_head)
        
        adapter_params = 0
        if self.use_early_adapter:
            adapter_params += sum(count(a) for a in self.early_adapters)
        if self.use_final_adapter:
            adapter_params += count(self.final_adapter)
        
        extras = router_params + early_head_params + adapter_params

        early_head_is_tied = any(
            h.weight is self.lm_head.weight or h.weight is self.model.embed_tokens.weight
            for h in self.early_lm_heads
        )
        tied_str = " (tied)" if early_head_is_tied else ""
        
        print_run("PARAMS", f"{'Component':<30} {'Params':>12} {'% of base':>12}")
        print_run("PARAMS", "-" * 56)
        print_run("PARAMS", f"{'Base model (backbone+lm_head)':<30} {base:>12,} {'100.00%':>12}")
        print_run("PARAMS", f"{'Routers ({len(self.router_layers)}x)':<30} {router_params:>12,} {100*router_params/base:>11.2f}%")
        print_run("PARAMS", f"{'Early heads{tied_str}':<30} {early_head_params:>12,} {100*early_head_params/base:>11.2f}%")
        print_run("PARAMS", f"{'Adapters':<30} {adapter_params:>12,} {100*adapter_params/base:>11.2f}%")
        print_run("PARAMS", "-" * 56)
        print_run("PARAMS", f"{'Total extras':<30} {extras:>12,} {100*extras/base:>11.2f}%")
        print_run("PARAMS", f"{'Total unique params':<30} {total_unique:>12,} {100*total_unique/base:>11.2f}%")

    def _random_init_all_parameters_like_pretraining(self):
        std = float(getattr(self.config, "initializer_range", 0.02))
        print_run("INIT", f"Re-initializing all parameters with std={std}...")
        def init_module(m: nn.Module):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if getattr(m, "padding_idx", None) is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif "RMSNorm" in m.__class__.__name__:
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
        self.apply(init_module)

        # Adapter init: fan_in-scaled input layer + zero-init output → starts as identity (h + 0 = h)
        if self.use_early_adapter:
            for adapter in self.early_adapters:
                fan_in = adapter[0].in_features
                nn.init.normal_(adapter[0].weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
                nn.init.zeros_(adapter[2].weight)
            print_run("INIT", "Early adapter layers initialized (scaled input + zero output).")

        if self.use_final_adapter:
            fan_in = self.final_adapter[0].in_features
            nn.init.normal_(self.final_adapter[0].weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
            nn.init.zeros_(self.final_adapter[2].weight)
            print_run("INIT", "Final adapter layers initialized (scaled input + zero output).")

    def forward(self, input_ids=None, attention_mask=None, labels=None, position_ids=None, 
                past_key_values=None, inputs_embeds=None, cache_position=None, use_cache=None, 
                current_step=0, **kwargs):

        # 1. Determine Training Phase & Active Configs
        is_warmup = (current_step < self.warmup_steps_limit)
        
        if is_warmup:
            active_flags = self.active_warmup
            alphas = self.alpha_map_warmup
        else:
            active_flags = self.active_main
            alphas = self.alpha_map_main

        # Prepare forward pass
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", True)
        if self.training:
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            self.config, inputs_embeds, attention_mask, cache_position,
            past_key_values=past_key_values, position_ids=position_ids
        )
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # Storage for multiple exits
        early_hidden_states = []
        early_logits_list = []
        router_logits_list = []
        
        exit_point_idx = 0
        
        # Layer loop with multiple exit points
        for i, decoder_layer in enumerate(self.model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            # Check if this is an exit point
            if exit_point_idx < len(self.early_layer_indices) and i == (self.early_layer_indices[exit_point_idx] - 1):
                early_hidden = hidden_states.clone()
                early_hidden_states.append(early_hidden)
                
                # Compute early logits with optional adapter
                e_h_norm = self.early_norms[exit_point_idx](early_hidden)
                
                if self.use_early_adapter:
                    adapted = e_h_norm + self.early_adapters[exit_point_idx](e_h_norm)
                    early_logits = self.early_lm_heads[exit_point_idx](adapted)
                else:
                    early_logits = self.early_lm_heads[exit_point_idx](e_h_norm)
                early_logits_list.append(early_logits)
                
                # Compute router decision
                router_input = e_h_norm
                router_outputs = self.router_layers[exit_point_idx](
                    router_input,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )[0]
                
                router_hidden = self.router_final_norms[exit_point_idx](router_outputs)
                router_logits = self.router_heads[exit_point_idx](router_hidden)
                router_logits_list.append(router_logits)

                if self.training and (current_step % 1000 == 0):
                    print_run("DEBUG", f"Step {current_step} | Exit {exit_point_idx} | ROUTER HIDDEN: {router_hidden.shape}")
                    print_run("DEBUG", f"Step {current_step} | Exit {exit_point_idx} | HIDDEN STATES: {hidden_states.shape}")
                    if self.use_early_adapter:
                        adapter_norm = self.early_adapters[exit_point_idx][2].weight.norm(2).item()
                        print_run("DEBUG", f"Step {current_step} | Exit {exit_point_idx} | Early adapter output norm: {adapter_norm:.6f}")
                    if self.use_final_adapter:
                        fa_norm = self.final_adapter[2].weight.norm(2).item()
                        print_run("DEBUG", f"Step {current_step} | Exit {exit_point_idx} | Final adapter output norm: {fa_norm:.6f}")
                
                exit_point_idx += 1

        # Final exit with optional adapter
        final_hidden = self.model.norm(hidden_states)
        if self.use_final_adapter:
            final_adapted = final_hidden + self.final_adapter(final_hidden)
            final_logits = self.lm_head(final_adapted)
        else:
            final_logits = self.lm_head(final_hidden)

        # Compute router weights & exit probablities
        router_weights_list = []
        exit_probabilities = []
        
        prob_reached = 1.0
        
        for router_logits in router_logits_list:
            router_probs = torch.softmax(router_logits / self.router_temp, dim=-1)
            p_exit = router_probs[..., 1:2]
            p_continue = router_probs[..., 0:1]
            
            router_weights = torch.cat([p_exit, p_continue], dim=-1).clamp(1e-6, 1.0 - 1e-6)
            router_weights_list.append(router_weights)
            
            total_p_exit = prob_reached * p_exit
            exit_probabilities.append(total_p_exit)
            
            prob_reached = prob_reached * p_continue
        
        exit_probabilities.append(prob_reached)
        exit_probs_tensor = torch.cat(exit_probabilities, dim=-1)

        # Inference vs training
        if labels is None:
            return self._inference_forward(
                final_logits, early_logits_list, exit_probs_tensor, attention_mask
            )

        # Training: compute losses
        all_logits = early_logits_list + [final_logits]
        shifted_logits = [lg[..., :-1, :].contiguous() for lg in all_logits]
        shifted_exit_probs = exit_probs_tensor[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        
        mask = (shifted_labels != -100)
        valid = mask & attention_mask[..., 1:].bool() if attention_mask is not None else mask
        
        safe_labels = shifted_labels.clone()
        safe_labels[~mask] = 0
        
        T = self.loss_temp
        
        log_probs_per_exit = []
        for logits in shifted_logits:
            logp = -F.cross_entropy(
                logits.view(-1, logits.size(-1)) / T, 
                safe_labels.view(-1), 
                reduction='none'
            ).view(logits.shape[:-1])
            log_probs_per_exit.append(logp)
        
        # --- Mixture loss ---
        weighted_probs = []
        for i, logp in enumerate(log_probs_per_exit):
            exit_prob = shifted_exit_probs[..., i].clamp(1e-8, 1.0)
            weighted_probs.append(torch.log(exit_prob) + logp)
        
        log_mix = torch.logsumexp(torch.stack(weighted_probs, dim=-1), dim=-1)
        val_mix_loss = -masked_mean(log_mix, valid)
        
        # --- Individual CE losses ---
        ce_losses = {}
        for i, logp in enumerate(log_probs_per_exit):
            ce_loss = -masked_mean(logp, valid)
            if i < len(log_probs_per_exit) - 1:
                ce_losses[f"early_ce_loss_{i+1}"] = ce_loss
            else:
                ce_losses["final_ce_loss"] = ce_loss
        
        # --- Compute loss ---
        router_compute_val = torch.tensor(0.0, device=final_logits.device)
        if active_flags.get("router_compute_loss", False):
            num_total_layers = len(self.model.layers)
            expected_compute = 0.0
            for i, layer_idx in enumerate(self.early_layer_indices):
                cost = layer_idx / num_total_layers
                exit_prob = shifted_exit_probs[..., i]
                expected_compute = expected_compute + (exit_prob * cost)
            cost_final = 1.0
            exit_prob_final = shifted_exit_probs[..., -1]
            expected_compute = expected_compute + (exit_prob_final * cost_final)
            
            router_compute_val = masked_mean(expected_compute, valid)
        
        # --- Router warmup loss ---
        router_warmup_val = torch.tensor(0.0, device=final_logits.device)
        if active_flags.get("router_warmup_loss", False):
            warmup_loss = 0.0
            for k, rw in enumerate(router_weights_list):
                p_exit_conditional = rw[..., :-1, 0]
                target_conditional = 1.0 / (self.num_exits - k)
                warmup_loss = warmup_loss + (p_exit_conditional - target_conditional).pow(2)
            router_warmup_val = masked_mean(warmup_loss, valid)
        
        
        # --- Diagnostics & logging ---
        dist_metrics = {}
        light_metrics = {}

        should_log_light = self.cfg.get("log_monitoring_light", True) and (current_step % self.light_interval == 0)
        should_log_heavy = self.cfg.get("log_monitoring_heavy", False) and (current_step % self.heavy_interval == 0)

        if should_log_light:
            with torch.no_grad():
                # Router norms
                router_norm_total = 0.0
                for router_layer, router_head in zip(self.router_layers, self.router_heads):
                    router_body_norm = sum(p.norm(2) for p in router_layer.parameters())
                    router_head_norm = router_head.weight.norm(2)
                    router_norm_total += (router_body_norm + router_head_norm)
                light_metrics["router/w_norm"] = router_norm_total.item()

                # Adapter norms
                if self.use_early_adapter:
                    for idx, adapter in enumerate(self.early_adapters):
                        light_metrics[f"adapter_norm/early_{idx}"] = adapter[2].weight.norm(2).item()
                if self.use_final_adapter:
                    light_metrics["adapter_norm/final"] = self.final_adapter[2].weight.norm(2).item()

                # Head entropies
                def get_avg_entropy_optimized(logits):
                    if logits.shape[0] == 0: return 0.0
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp()
                    entropy = -(probs * log_probs).sum(dim=-1)
                    return entropy.mean().item()

                for i, logits in enumerate(shifted_logits):
                    valid_logits = logits[valid]
                    if valid_logits.shape[0] > 0:
                        if i < len(shifted_logits) - 1:
                            light_metrics[f"entropy/early_{i+1}"] = get_avg_entropy_optimized(valid_logits)
                        else:
                            light_metrics["entropy/final"] = get_avg_entropy_optimized(valid_logits)

                # Exit probabilities
                for i in range(self.num_exits):
                    mean_exit_prob = masked_mean(shifted_exit_probs[..., i], valid)
                    if i < self.num_exits - 1:
                        light_metrics[f"exit_prob/early_{i+1}"] = mean_exit_prob.item()
                    else:
                        light_metrics["exit_prob/final"] = mean_exit_prob.item()
                
                # Heavy metrics
                if should_log_heavy:
                    for i in range(len(shifted_logits) - 1):
                        valid_early = shifted_logits[i][valid]
                        valid_final = shifted_logits[-1][valid]
                        if valid_early.shape[0] > 0:
                            metrics = compute_distribution_metrics(
                                valid_early, valid_final, k=5, p=0.95
                            )
                            for key, val in metrics.items():
                                dist_metrics[f"head_{i+1}_{key}"] = val
        
        # Build loss dict
        loss_dict = {
            "mix_loss": val_mix_loss,
            **ce_losses,
            "router_compute_loss": router_compute_val,
            "router_warmup_loss": router_warmup_val,
            **{f"heads/{k}": v for k, v in dist_metrics.items()},
            **light_metrics
        }
        
        # Aggregate losses
        total_loss = torch.tensor(0.0, device=final_logits.device)
        for key, val in loss_dict.items():
            w = alphas.get(key, 0.0)
            
            if w > 0.0:
                total_loss = total_loss + (w * val)
        
        return CustomModelOutput(
            loss=total_loss,
            logits=final_logits,
            hidden_states=tuple(early_hidden_states + [final_hidden]),
            loss_dict=loss_dict,
            router_weights=router_weights_list,
            exit_probabilities=exit_probs_tensor
        )

    def _inference_forward(self, final_logits, early_logits_list, exit_probs_tensor, attention_mask):
        # Simulate inference-time decoding behaviour based on decode_head strategy
        if self.decode_head == "mix":
            all_logits = early_logits_list + [final_logits]
            all_log_probs = [F.log_softmax(lg, dim=-1) for lg in all_logits]
            
            weighted_log_probs = []
            for i, log_prob in enumerate(all_log_probs):
                exit_prob = exit_probs_tensor[..., i:i+1]
                weighted_log_probs.append(exit_prob.log() + log_prob)
            
            logits = torch.logsumexp(torch.stack(weighted_log_probs, dim=-1), dim=-1)
            
        elif self.decode_head.startswith("early_"):
            head_idx = int(self.decode_head.split("_")[1]) - 1
            logits = early_logits_list[head_idx]
            
        elif self.decode_head == "router_sample":
            B, L, num_exits = exit_probs_tensor.shape
            flat_idx = torch.multinomial(exit_probs_tensor.view(B * L, num_exits), num_samples=1)
            exit_idx = flat_idx.view(B, L)  # [B, L]
            all_logits = early_logits_list + [final_logits]
            stacked = torch.stack(all_logits, dim=0)  # [num_exits, B, L, vocab]
            vocab = stacked.shape[-1]
            idx = exit_idx.unsqueeze(0).unsqueeze(-1).expand(1, B, L, vocab)
            logits = torch.gather(stacked, dim=0, index=idx).squeeze(0)  # [B, L, vocab]
            
        else:  # "final" or default
            logits = final_logits
        
        return CausalLMOutputWithPast(
            loss=None, 
            logits=logits, 
            past_key_values=None
        )

    def generate(self, *args, **kwargs):
        if hasattr(self, "_eval_gen_kwargs"):
            for k, v in self._eval_gen_kwargs.items():
                kwargs[k] = v
        return super().generate(*args, **kwargs)