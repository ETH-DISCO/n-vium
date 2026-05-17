# calm_main.py
#
# Training entry-point for CALM.
#
# Supports two modes:
#   Pretraining (baseline_checkpoint: null):
#     Random init from model_name_or_path config + model_overrides.
#     Full Chinchilla budget; standard cosine schedule.
#   Continued pretraining (baseline_checkpoint: <path>):
#     Warm-start from a dense baseline; exit norms synced from model.norm.
#     ~25% budget; cold-restart cosine schedule.

import os
import sys
import json
import torch
import wandb
import shutil

from calm_model import CalmModel
from calm_trainer import CalmTrainer
from transformers.integrations import WandbCallback
from transformers import set_seed

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    AutoConfig,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Nvium"))
from Nvium_utils import get_dataset, print_run, print_gpu_debug, TokensPerSecondCallback, CustomWandbCallback

import hydra
from omegaconf import OmegaConf


#######################################################################
# DDP + HYDRA SAFETY
#######################################################################

rank = int(os.environ.get("RANK", "0"))
if rank != 0:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HYDRA_MAIN_PROCESS_ONLY"] = "1"

cache_dir = os.path.join(os.environ.get("SCRATCH", "/tmp"), "cache")
os.makedirs(cache_dir, exist_ok=True)

print_run("INFO", f"Using cache dir: {cache_dir}")
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_TOKENIZERS_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir


#######################################################################
# BUILD MODEL
#######################################################################

def get_dtype(load_args):
    if load_args.get("bf16", False):
        return torch.bfloat16
    if load_args.get("fp16", False):
        return torch.float16
    return torch.float32


def build_model(exp_cfg):
    """
    Build a CalmModel.

    Pretraining (baseline_checkpoint absent / null):
        Random init from model_name_or_path config + model_overrides.
        Per-exit RMSNorms are deepcopied from the randomly-init'd model.norm
        in __init__ — no re-sync needed.

    Continued pretraining (baseline_checkpoint provided):
        Loads backbone weights from the dense baseline checkpoint, then
        re-syncs per-exit RMSNorms from the loaded model.norm.
    """
    load_args = exp_cfg["load_args"]
    torch_dtype = get_dtype(load_args)
    attn_impl = load_args.get("attn_implementation", None)
    model_overrides = exp_cfg.get("model_overrides", {})
    baseline_ckpt = exp_cfg.get("baseline_checkpoint") or None

    # --- Config -----------------------------------------------------------
    config_source = baseline_ckpt if baseline_ckpt else exp_cfg["model_name_or_path"]
    hf_config = AutoConfig.from_pretrained(
        config_source,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=bool(baseline_ckpt),
    )
    if model_overrides:
        print_run("CONFIG", f"Applying model_overrides: {model_overrides}")
        hf_config.update(model_overrides)

    model = CalmModel(
        hf_config,
        exp_config=exp_cfg,
        attn_implementation=attn_impl,
    )

    # --- Weights ----------------------------------------------------------
    if baseline_ckpt:
        from safetensors.torch import load_file as sf_load
        import glob as _glob

        print_run("MODEL", f"Loading dense baseline from: {baseline_ckpt}")
        ckpt_files = _glob.glob(os.path.join(baseline_ckpt, "*.safetensors"))
        if not ckpt_files:
            raise FileNotFoundError(f"No .safetensors files found in {baseline_ckpt}")

        state_dict = {}
        for f in sorted(ckpt_files):
            state_dict.update(sf_load(f))
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        calm_missing = [k for k in missing if "exit_norms" not in k]
        if calm_missing:
            print_run("RESUME", f"Unexpected missing keys: {calm_missing}")
        if unexpected:
            print_run("RESUME", f"Unexpected keys: {unexpected}")
        print_run("MODEL", f"Loaded {len(cleaned)} backbone keys from {baseline_ckpt}")

        # Re-sync per-exit RMSNorms: deepcopy in __init__ captured random
        # init; must re-copy from the now-loaded model.norm weights.
        if getattr(model, "per_exit_norms_enabled", False):
            with torch.no_grad():
                src_w = model.model.norm.weight.data
                for idx in model.exit_layer_indices:
                    model.exit_norms[str(idx)].weight.data.copy_(src_w)
            print_run("MODEL", f"Per-exit RMSNorms synced from model.norm")
    else:
        print_run("MODEL", "Random init — pretraining from scratch.")

    if not getattr(model, "per_exit_norms_enabled", False):
        print_run("MODEL", "Using shared model.norm at every exit (paper-default).")

    model = model.to(dtype=torch_dtype)
    model.print_parameter_summary()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print_run("INFO", f"Trainable: {trainable} / {total} ({100*trainable/total:.4f}%)")

    json_str = json.dumps(hf_config.to_dict(), indent=4)
    indented = "\n".join(["    " + l for l in json_str.splitlines()])
    print_run("CONFIG", f"HF config:\n{indented}")

    return model


#######################################################################
# MAIN
#######################################################################

@hydra.main(version_base=None, config_path="../../config", config_name="calm_config")
def main(cfg):
    set_seed(42)
    print_run("INFO", "Global seed set to 42")

    run_dir = os.getcwd()
    print_run("HYDRA", "Hydra run dir:", run_dir)

    stdout_path = os.path.join(run_dir, "stdout.log")
    stderr_path = os.path.join(run_dir, "stderr.log")
    sys.stdout = open(stdout_path, "a")
    sys.stderr = open(stderr_path, "a")

    exp_config = OmegaConf.to_container(cfg, resolve=True)
    baseline_ckpt = exp_config.get("baseline_checkpoint") or None
    base_model_name = exp_config["model_name_or_path"]

    wandb_dir = os.path.join(run_dir, "wandb")
    os.environ["WANDB_DIR"] = wandb_dir
    os.makedirs(wandb_dir, exist_ok=True)

    src_folder = os.path.dirname(__file__)
    dst_folder = os.path.join(run_dir, "source_snapshot")
    print_run("INFO", f"Copying source → {dst_folder}")
    shutil.copytree(
        src_folder, dst_folder, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".git", "wandb", "outputs"),
    )

    yaml_str = OmegaConf.to_yaml(cfg)
    indented = "\n".join(["    " + l for l in yaml_str.splitlines()])
    print_run("CONFIG", f"Config:\n{indented}")

    print_run("CUDA", "torch.cuda.is_available():", torch.cuda.is_available())
    print_run("CUDA", "cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print_run("CUDA", "Current device:", torch.cuda.current_device())
        print_run("CUDA", "Device name:", torch.cuda.get_device_name(0))
    print_gpu_debug(prefix="GPU-INIT")

    ###################################################################
    # Tokenizer
    ###################################################################
    tokenizer_path = exp_config.get("tokenizer_path") or baseline_ckpt or base_model_name
    local_only = not tokenizer_path.startswith("JackFram") and not tokenizer_path.startswith("meta-llama")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###################################################################
    # Model
    ###################################################################
    model = build_model(exp_config)
    print_run("ATTENTION", "flash_sdp enabled:", torch.backends.cuda.flash_sdp_enabled())
    print_run("ATTENTION", "mem_efficient_sdp enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())

    ###################################################################
    # Dataset
    ###################################################################
    tok = get_dataset(exp_config, cache_dir, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ###################################################################
    # Dynamic step calculations
    ###################################################################
    ta_cfg = cfg.training_args
    output_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    # Save resolved config alongside checkpoints for traceability
    config_save_path = os.path.join(output_dir, "train_config.yaml")
    with open(config_save_path, "w") as _f:
        _f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print_run("CONFIG", f"Saved resolved config to {config_save_path}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = (
        ta_cfg.per_device_train_batch_size
        * ta_cfg.gradient_accumulation_steps
        * world_size
    )
    num_training_sequences = len(tok["train"])
    epoch_steps = (num_training_sequences // global_batch_size) * ta_cfg.num_train_epochs
    cfg_max_steps = int(getattr(ta_cfg, "max_steps", -1))
    total_optimization_steps = cfg_max_steps if cfg_max_steps > 0 else epoch_steps
    save_interval = max(ta_cfg.save_steps, int(total_optimization_steps * ta_cfg.eval_save_interval_percent))

    print_run("INFO", f"Global Batch Size:          {global_batch_size}")
    print_run("INFO", f"Total Optimization Steps:   {total_optimization_steps}")
    print_run("INFO", f"Dynamic Eval/Save Interval: {save_interval}")

    ###################################################################
    # TrainingArguments
    ###################################################################
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=ta_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=ta_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=ta_cfg.gradient_accumulation_steps,
        gradient_checkpointing=ta_cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=ta_cfg.gradient_checkpointing_kwargs,
        learning_rate=float(ta_cfg.learning_rate),
        num_train_epochs=ta_cfg.num_train_epochs,
        max_steps=cfg_max_steps,
        optim=ta_cfg.optim,
        adam_beta1=ta_cfg.adam_beta1,
        adam_beta2=ta_cfg.adam_beta2,
        adam_epsilon=float(ta_cfg.adam_epsilon),
        lr_scheduler_type=ta_cfg.lr_scheduler_type,
        warmup_ratio=ta_cfg.warmup_ratio,
        lr_scheduler_kwargs={"min_lr": ta_cfg.min_lr},
        max_grad_norm=ta_cfg.max_grad_norm,
        weight_decay=ta_cfg.weight_decay,
        fp16=ta_cfg.fp16,
        bf16=ta_cfg.bf16,
        dataloader_num_workers=ta_cfg.dataloader_num_workers,
        dataloader_pin_memory=ta_cfg.dataloader_pin_memory,
        dataloader_prefetch_factor=ta_cfg.dataloader_prefetch_factor,
        logging_steps=ta_cfg.logging_steps,
        save_steps=save_interval,
        eval_steps=save_interval,
        eval_strategy=ta_cfg.eval_strategy,
        report_to=ta_cfg.report_to,
        run_name=ta_cfg.run_name,
        torch_compile=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,  # exit norms always used
    )

    ###################################################################
    # Trainer
    ###################################################################
    seq_len = exp_config.get("model_overrides", {}).get("max_position_embeddings", 1024)

    trainer = CalmTrainer(
        model=model,
        args=args,
        train_dataset=tok["train"],
        eval_dataset=tok.get("validation"),
        data_collator=collator,
        callbacks=[
            TokensPerSecondCallback(seq_len=seq_len, global_batch_size=global_batch_size),
        ],
    )

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank <= 0:
        trainer.remove_callback(WandbCallback)
        trainer.add_callback(CustomWandbCallback())
        if wandb.run is None:
            wandb.init(
                project=ta_cfg.project_name,
                name=args.run_name,
                settings=wandb.Settings(start_method="thread"),
            )
    else:
        trainer.remove_callback(WandbCallback)

    resume_ckpt = exp_config.get("resume_from_checkpoint") or None
    if resume_ckpt:
        import os as _os
        if not _os.path.exists(_os.path.join(resume_ckpt, "optimizer.pt")):
            print_run("RESUME", f"No optimizer.pt in {resume_ckpt} — starting fresh.")
            resume_ckpt = None
        else:
            print_run("RESUME", f"Resuming from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)


if __name__ == "__main__":
    main()
