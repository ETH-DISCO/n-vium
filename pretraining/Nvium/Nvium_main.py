import os
import sys
import json
import torch
import wandb
import shutil


from Nvium_utils import get_dataset, print_run, print_gpu_debug, TokensPerSecondCallback, CustomWandbCallback
from Nvium_model import NviumModel
from Nvium_baseline import BaselineLlamaModel
from Nvium_trainer import ForwardTrainer, BaselineCESaverCallback, AdaptivePenaltyCallback
from transformers.integrations import WandbCallback
from transformers import set_seed

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

import hydra
from omegaconf import OmegaConf


#######################################################################
# DDP + HYDRA SAFETY
#######################################################################

# only rank 0 should perform Hydra directory creation
rank = int(os.environ.get("RANK", "0"))
if rank != 0:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HYDRA_MAIN_PROCESS_ONLY"] = "1"

SCRATCH = os.environ.get("SCRATCH")
if not SCRATCH:
    raise RuntimeError("SCRATCH is not set")
cache_dir = os.path.join(SCRATCH, "cache")
os.makedirs(cache_dir, exist_ok=True)

print_run("INFO", f"Using cache dir: {cache_dir}")
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_TOKENIZERS_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir


#######################################################################
# BUILD MODEL FUNCTION
#######################################################################
def get_dtype(load_args):
    if load_args.get("bf16", False):
        return torch.bfloat16
    if load_args.get("fp16", False):
        return torch.float16
    return torch.float32

def build_model(exp_cfg):
    base_model_name = exp_cfg["model_name_or_path"]
    ckpt_path = exp_cfg.get("init_checkpoint", None)

    load_args = exp_cfg["load_args"]
    torch_dtype = get_dtype(load_args)
    attn_impl = load_args.get("attn_implementation", None)

    # ---------------------------------------------------------
    # DYNAMIC MODEL SELECTION
    # ---------------------------------------------------------
    model_type = exp_cfg.get("model_type", "Nvium")

    if model_type == "baseline":
        print_run("MODEL", "Architecture selected: BaselineLlamaModel")
        ModelClass = BaselineLlamaModel
    elif model_type == "Nvium":
        print_run("MODEL", "Architecture selected: NviumModel")
        ModelClass = NviumModel
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'baseline' or 'Nvium'.")

    # Always load HF config for prints + architecture
    hf_config = AutoConfig.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=False,
    )

    # Apply dynamic config overrides
    model_overrides = exp_cfg.get("model_overrides", {})
    if model_overrides:
        print_run("CONFIG", f"Applying dynamic config overrides: {model_overrides}")
        hf_config.update(model_overrides)

    # ----------------------------
    # Build model (two paths)
    # ----------------------------
    if exp_cfg.get("random_init_all", False):
        print_run("MODEL", f"Building model from config (scratch init): {base_model_name}")

        model = ModelClass(
            hf_config,
            exp_config=exp_cfg,
            attn_implementation=attn_impl,
        )

        # Put model in desired dtype (optional; Trainer autocast may handle bf16)
        model = model.to(dtype=torch_dtype)

    else:
        # Load weights from checkpoint or base
        load_path = ckpt_path if ckpt_path else base_model_name
        print_run("MODEL", f"Loading model weights from: {load_path}")

        model = ModelClass.from_pretrained(
            load_path,
            config=hf_config,
            exp_config=exp_cfg,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )

        # Re-apply tying (loading weights can break shared pointers)
        tie_cfg = exp_cfg.get("tying", {})
        if hasattr(model, "_apply_tying"):
            model._apply_tying(tie_cfg)
            print_run("FIX", "Re-applied head tying (pointers shared) after weight load.")
        
        # Reset router parameters
        if hasattr(model, "reset_all_router_parameters"):
            model.reset_all_router_parameters()
            print_run("FIX", "Reset all router parameters after weight load.")

    # ----------------------------
    # Prints
    # ----------------------------
    print_run("INFO", "New Model Trainable Parameter Count: ", model.num_parameters(only_trainable=True))
    print_run("INFO", "New Model Total Parameter Count: ", model.num_parameters(only_trainable=False))
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print_run("INFO", f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.4f}%)")
    
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    print_run("INFO", f"Trainable Parameters List: {trainable_names}")
    
    json_str = json.dumps(hf_config.to_dict(), indent=4)
    indented_json = "\n".join(["    " + line for line in json_str.splitlines()])
    print_run("CONFIG", f"Loaded HF config:\n{indented_json}")

    print_run("INFO", "Model type:", hf_config.model_type)
    print_run("INFO", "Experiment config loaded.")
    
    first_layer = model.model.layers[0]
    attn = getattr(first_layer, "self_attn", None)
    attn0 = model.model.layers[0].self_attn
    
    print_run("INFO", "self_attn class:", type(attn0))
    print_run("INFO", "self_attn module:", attn0)
    print_run("INFO", "self_attn forward file:", attn0.forward.__code__.co_filename)
    print_run("INFO", " self_attn class:", attn.__class__.__name__ if attn is not None else "None")
    print_run("INFO", "config._attn_implementation:", getattr(model.config, "_attn_implementation", None))
    print_run("INFO", "Loaded precision", torch_dtype)

    return model


#######################################################################
# MAIN TRAINING FUNCTION
#######################################################################

@hydra.main(version_base=None, config_path="../../config", config_name="Nvium_config")
def main(cfg):
    seed = int(cfg.seed)
    set_seed(seed)
    print_run("INFO", "Global seed set to ", seed)
    ###################################################################
    # Hydra run directory
    ###################################################################
    run_dir = os.getcwd()
    print_run("HYDRA", "Hydra run dir:", run_dir)

    # Redirect stdout & stderr to files inside Hydra output directory
    stdout_path = os.path.join(run_dir, "stdout.log")
    stderr_path = os.path.join(run_dir, "stderr.log")

    sys.stdout = open(stdout_path, "a")
    sys.stderr = open(stderr_path, "a")

    print_run("HYDRA", f"[Hydra] stdout redirected to {stdout_path}")
    print_run("HYDRA", f"[Hydra] stderr redirected to {stderr_path}", file=sys.stderr)

    exp_config = OmegaConf.to_container(cfg, resolve=True)
    base_model_name = exp_config["model_name_or_path"]

    ###################################################################
    # Isolate wandb inside Hydra folder
    ###################################################################
    wandb_dir = os.path.join(run_dir, "wandb")
    os.environ["WANDB_DIR"] = wandb_dir
    os.makedirs(wandb_dir, exist_ok=True)

    ###################################################################
    # Copy source for reproducibility
    ###################################################################
    src_folder = os.path.dirname(__file__)
    dst_folder = os.path.join(run_dir, "source_snapshot")

    print_run("INFO", f"Copying source code → {dst_folder}")

    shutil.copytree(
        src_folder,
        dst_folder,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo",
            ".git", ".idea", "wandb", "outputs"
        )
    )

    ###################################################################
    # CUDA INFO
    ###################################################################
    yaml_str = OmegaConf.to_yaml(cfg)
    indented_yaml = "\n".join(["    " + line for line in yaml_str.splitlines()])
    print_run("CONFIG", f"Config:\n{indented_yaml}")

    print_run("CUDA", "torch.cuda.is_available():", torch.cuda.is_available())
    print_run("CUDA", "cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print_run("CUDA", "Current device:", torch.cuda.current_device())
        print_run("CUDA", "Device name:", torch.cuda.get_device_name(0))
    print_gpu_debug(prefix="GPU-INIT")

    ###################################################################
    # Tokenizer
    ###################################################################
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        print_run("INFO", "Setting pad token to eos token...")
        tokenizer.pad_token = tokenizer.eos_token

    ###################################################################
    # Load model
    ###################################################################
    model = build_model(exp_config)
    print_run("ATTENTION", "flash_sdp enabled:", torch.backends.cuda.flash_sdp_enabled())
    print_run("ATTENTION", "mem_efficient_sdp enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print_run("ATTENTION", "math_sdp enabled:", torch.backends.cuda.math_sdp_enabled())

    if exp_config.get("compile", {}).get("torch_compile", False):
        print_run("COMPILE", "Compiling model with torch.compile()...")
        model = torch.compile(
            model, 
            options={
                "max_autotune": True,
                "triton.cudagraphs": False,
                "epilogue_fusion": True,
                "shape_padding": True,
                "coordinate_descent_tuning": True,
            }
        )

    ###################################################################
    # Dataset + collator
    ###################################################################
    tok = get_dataset(exp_config, cache_dir, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None)

    ###################################################################
    # Dynamic Step Calculations
    ###################################################################
    ta_cfg = cfg.training_args
    output_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    # Save resolved config alongside checkpoints for traceability
    config_save_path = os.path.join(output_dir, "train_config.yaml")
    with open(config_save_path, "w") as _f:
        _f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print_run("CONFIG", f"Saved resolved config to {config_save_path}")

    # 1. Dynamically find how many GPUs you are using (defaults to 1 if not distributed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 2. Calculate true global batch size
    global_batch_size = (
        ta_cfg.per_device_train_batch_size
        * ta_cfg.gradient_accumulation_steps
        * world_size
    )
    
    # 3. Calculate total optimization steps for the entire run
    num_training_sequences = len(tok["train"])
    epoch_steps = (num_training_sequences // global_batch_size) * ta_cfg.num_train_epochs
    cfg_max_steps = int(getattr(ta_cfg, "max_steps", -1))
    total_optimization_steps = cfg_max_steps if cfg_max_steps > 0 else epoch_steps

    # 4. Calculate save/eval interval (percentage of total steps, with minimum)
    save_interval = max(ta_cfg.save_steps, int(total_optimization_steps * ta_cfg.eval_save_interval_percent))

    print_run("INFO", f"Calculated Global Batch Size:  {global_batch_size} sequences")
    print_run("INFO", f"Total Optimization Steps:      {total_optimization_steps}" + (" (from max_steps)" if cfg_max_steps > 0 else " (from dataset)"))
    print_run("INFO", f"Dynamic Eval/Save Interval:    {save_interval} steps")

    # LR scheduler kwargs — WSD needs stable/decay step counts computed at runtime
    if ta_cfg.lr_scheduler_type == "warmup_stable_decay":
        _warmup_steps = int(total_optimization_steps * ta_cfg.warmup_ratio)
        _decay_steps = int(total_optimization_steps * getattr(ta_cfg, "lr_decay_ratio", 0.1))
        _stable_steps = total_optimization_steps - _warmup_steps - _decay_steps
        _min_lr_ratio = float(ta_cfg.min_lr) / float(ta_cfg.learning_rate)
        _lr_scheduler_kwargs = {
            "num_stable_steps": _stable_steps,
            "num_decay_steps": _decay_steps,
            "min_lr_ratio": _min_lr_ratio,
        }
        print_run("LR", f"WSD schedule: warmup={_warmup_steps} stable={_stable_steps} decay={_decay_steps} min_lr_ratio={_min_lr_ratio:.3f}")
    else:
        _lr_scheduler_kwargs = {"min_lr": ta_cfg.min_lr}

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
        lr_scheduler_kwargs=_lr_scheduler_kwargs,
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

        # Torch compile options
        torch_compile=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
    )

    ###################################################################
    # Trainer
    ###################################################################
    seq_len = exp_config.get("model_overrides", {}).get("max_position_embeddings", 1024)

    trainer = ForwardTrainer(
        model=model,
        args=args,
        train_dataset=tok["train"],
        eval_dataset=tok.get("validation"),
        data_collator=collator,
        callbacks=[
            TokensPerSecondCallback(seq_len=seq_len, global_batch_size=global_batch_size),
        ],
    )

    # Inject loss warmup into the model dynamically
    loss_warmup_ratio = exp_config.get("loss", {}).get("warmup_ratio", 0.05)
    fixed_warmup_steps = exp_config.get("loss", {}).get("warmup_steps", 0)
    if hasattr(model, "warmup_steps_limit"):
        model.warmup_steps_limit = max(fixed_warmup_steps, int(total_optimization_steps * loss_warmup_ratio))
        print_run("LOSS", f"Set model.warmup_steps_limit to {model.warmup_steps_limit} steps ({loss_warmup_ratio*100:.1f}% of total)")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Only remove/add callbacks and init wandb on the main process
    if local_rank <= 0:  # -1 for single GPU, 0 for main process in DDP
        trainer.remove_callback(WandbCallback)
        trainer.add_callback(CustomWandbCallback())
        
        if wandb.run is None:
            wandb.init(
                project=ta_cfg.project_name,
                name=args.run_name,
                settings=wandb.Settings(start_method="thread") 
            )
    else:
        trainer.remove_callback(WandbCallback)

    # Baseline CE saver — attach to baseline runs to produce the quality-target JSON
    if getattr(ta_cfg, "save_baseline_ce", False):
        baseline_ce_path = os.path.join(output_dir, "baseline_ce_records.json")
        _saver_cb = BaselineCESaverCallback(baseline_ce_path)
        _saver_cb._trainer_ref = trainer
        trainer.add_callback(_saver_cb)
        print_run("ADAPTIVE", f"BaselineCESaverCallback → {baseline_ce_path}")

    # Adaptive compute penalty — attach to Nvium runs with a baseline JSON
    _adaptive_cfg = exp_config.get("adaptive_penalty", {}) or {}
    if _adaptive_cfg.get("enabled", False):
        _baseline_ce_path = _adaptive_cfg.get("baseline_ce_path", "")
        if not _baseline_ce_path or not os.path.exists(_baseline_ce_path):
            raise FileNotFoundError(
                f"adaptive_penalty.enabled=true but baseline_ce_path not found: '{_baseline_ce_path}'"
            )
        _adaptive_cb = AdaptivePenaltyCallback(
            model=model,
            baseline_ce_path=_baseline_ce_path,
            k_up=float(_adaptive_cfg.get("k_up", 2.0)),
            k_down=float(_adaptive_cfg.get("k_down", 4.0)),
            dead_band=float(_adaptive_cfg.get("dead_band", 0.005)),
            max_step=float(_adaptive_cfg.get("max_step", 0.2)),
            min_weight=float(_adaptive_cfg.get("min_weight", 0.0)),
            max_weight=float(_adaptive_cfg.get("max_weight", 2.0)),
        )
        _adaptive_cb._trainer_ref = trainer
        trainer.add_callback(_adaptive_cb)
        print_run("ADAPTIVE", f"AdaptivePenaltyCallback will read baseline from {_baseline_ce_path} "
                              f"k_up={_adaptive_cfg.get('k_up', 2.0)} k_down={_adaptive_cfg.get('k_down', 4.0)} "
                              f"dead_band={_adaptive_cfg.get('dead_band', 0.005)} max_step={_adaptive_cfg.get('max_step', 0.2)} "
                              f"weight_range=[{_adaptive_cfg.get('min_weight', 0.0)}, {_adaptive_cfg.get('max_weight', 2.0)}]")

    resume_ckpt = exp_config.get("resume_from_checkpoint", None)
    init_ckpt = exp_config.get("init_checkpoint", None)
    # If init_checkpoint was used, from_pretrained already loaded the weights.
    # Only do the manual load when resuming from a path that wasn't used for init.
    if resume_ckpt and not init_ckpt:
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(resume_ckpt, "model.safetensors"))
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print_run("RESUME", f"Missing keys: {missing}")
        if unexpected:
            print_run("RESUME", f"Unexpected keys: {unexpected}")
        print_run("RESUME", f"Loaded model weights from {resume_ckpt} ({len(cleaned)} keys)")
    # Only pass resume_from_checkpoint to Trainer if it's a full Trainer checkpoint
    # (has optimizer/scheduler state). A bare model checkpoint (only model.safetensors)
    # has no optimizer state, so we pass None and start fresh.
    trainer_ckpt = resume_ckpt if (resume_ckpt and os.path.exists(os.path.join(resume_ckpt, "optimizer.pt"))) else None
    trainer.train(resume_from_checkpoint=trainer_ckpt)


if __name__ == "__main__":
    main()