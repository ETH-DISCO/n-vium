import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache

def get_baseline_model_class(BaseModelClass):
    """
    Wraps the custom model to force a standard (dense) forward pass 
    using the OFFICIAL backbone implementation to prevent RoPE errors.
    """
    class InferenceBaselineModel(BaseModelClass):
        def __init__(self, config, exp_config=None, **kwargs):
            if exp_config is None:
                exp_config = kwargs.pop("exp_config", None)
            
            # Initialize parent (loads weights, etc.)
            super().__init__(config, exp_config=exp_config, **kwargs)
            
            # API Compatibility for Benchmark Script
            self.history = {"decisions": [], "layers_skipped": []}
            self.deferred_queue = [] 

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True, **kwargs):
            # 1. Setup
            batch_size, seq_len = input_ids.shape
            
            # 2. RUN OFFICIAL BACKBONE (Fixes RoPE/Shape errors)
            # Instead of manually looping layers (which caused your crash),
            # we invoke the LlamaModel forward pass directly.
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )
            
            hidden_states = outputs[0]
            past_key_values = outputs.past_key_values

            # 3. Final Head
            logits = self.lm_head(hidden_states)
            
            # 4. Track Stats (Always "Final", 0 skipped)
            # We only track during decoding (seq_len == 1) to match the Optimized logic
            if seq_len == 1:
                self.history["decisions"].append("final")
                self.history["layers_skipped"].append(0)

            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=past_key_values,
            )

        @torch.no_grad()
        def generate_with_stats(self, input_ids, **kwargs):
            # Reset stats before generation
            self.history = {"decisions": [], "layers_skipped": []}
            self.deferred_queue = [] 
            return self.generate(input_ids, **kwargs)

    return InferenceBaselineModel