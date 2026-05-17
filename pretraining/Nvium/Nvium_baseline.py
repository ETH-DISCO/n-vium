import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
from dataclasses import dataclass
from Nvium_utils import print_run

@dataclass
class CustomModelOutput(CausalLMOutputWithPast):
    loss_dict: Optional[dict] = None
    router_weights: Optional[torch.FloatTensor] = None

class BaselineLlamaModel(LlamaForCausalLM):
    def __init__(self, config, exp_config=None, **kwargs):
        attn_impl = kwargs.pop("attn_implementation", None)
        super().__init__(config, **kwargs)
        if attn_impl is not None:
            self.config._attn_implementation = attn_impl

        self.cfg = exp_config if exp_config is not None else {}

        if self.cfg.get("random_init_all", True):
            self._random_init_all_parameters_like_pretraining()

        self._apply_tying(self.cfg.get("tying", {}))

        self.register_buffer("_step_tensor", torch.tensor(0, dtype=torch.long))

    @property
    def _global_step(self):
        return self._step_tensor.item()

    @_global_step.setter
    def _global_step(self, value):
        self._step_tensor.fill_(value)

    def _apply_tying(self, tie_cfg: dict):
        tie_final = tie_cfg.get("tie_final_to_embeddings", True)
        if tie_final:
            self.lm_head.weight = self.model.embed_tokens.weight
            self.config.tie_word_embeddings = True
            print_run("INIT", "Tied final LM head to word embeddings.")
        else:
            if self.lm_head.weight is self.model.embed_tokens.weight:
                self.lm_head.weight = nn.Parameter(self.lm_head.weight.detach().clone())
            self.config.tie_word_embeddings = False
            print_run("INIT", "Untied final LM head from word embeddings.")

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

    def forward(self, input_ids=None, attention_mask=None, labels=None, position_ids=None, past_key_values=None, inputs_embeds=None, cache_position=None, use_cache=None, **kwargs):

        if self.training:
            self._step_tensor += 1

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )

        loss_dict = {}
        if labels is not None and outputs.loss is not None:
            loss_dict = {
                "final_ce_loss": outputs.loss,
            }

        return CustomModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_dict=loss_dict,
            router_weights=None
        )

    def generate(self, *args, **kwargs):
        if hasattr(self, "_eval_gen_kwargs"):
            for k, v in self._eval_gen_kwargs.items():
                kwargs[k] = v
        return super().generate(*args, **kwargs)
