import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-3B")
    block_size: int = field(default=20000)
    wandb_project: Optional[str] = field(default="s1")
    wandb_entity: Optional[str] = field(default="hashimoto-group")
    use_wandb: bool = field(default=False)
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    results_dir: str = field(default="results")

    def __post_init__(self):
        if self.use_wandb:
            os.environ['WANDB_PROJECT'] = self.wandb_project
            os.environ['WANDB_ENTITY'] = self.wandb_entity
        else:
            # Disable wandb to avoid accidental online logging
            os.environ['WANDB_MODE'] = 'disabled'


def _save_training_plots(trainer, output_dir: str):
    try:
        logs = getattr(trainer.state, "log_history", None) or []
    except Exception:
        logging.exception("Could not read trainer.state.log_history")
        logs = []

    if not logs:
        logging.info("No training logs found to plot.")
        return

    results_path = Path(output_dir or "results")
    results_path.mkdir(parents=True, exist_ok=True)

    traces = defaultdict(list)
    for i, entry in enumerate(logs):
        step = entry.get('step', None)
        step_idx = step if step is not None else i
        for k, v in entry.items():
            if k in ("step", "epoch", "total_flos", "train_runtime", "timestamp"):
                continue
            if isinstance(v, (int, float)):
                traces[k].append((step_idx, float(v)))

    # Save raw parsed metrics
    serializable = {k: [[int(x), float(y)] for x, y in vals] for k, vals in traces.items()}
    with open(results_path / "metrics.json", "w") as fh:
        json.dump(serializable, fh, indent=2)

    # Plot each numeric metric
    for metric, vals in traces.items():
        xs = [x for x, _ in vals]
        ys = [y for _, y in vals]
        if len(xs) == 0:
            continue
        plt.figure()
        plt.plot(xs, ys, marker='o' if len(xs) < 50 else None)
        plt.xlabel('step')
        plt.ylabel(metric)
        plt.title(metric)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_path / f"{metric}.png")
        plt.close()

    logging.info(f"Saved metrics plots to {results_path}")

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    # Reduce memory by disabling cache and enabling gradient checkpointing when available.
    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception:
        logging.exception("Failed to set model.config.use_cache = False")

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            logging.info("Enabled model.gradient_checkpointing_enable()")
        except Exception:
            logging.exception("Failed to enable gradient checkpointing on model")

    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    try:
        out_dir = getattr(args, 'output_dir', None) or config.results_dir
        _save_training_plots(trainer, out_dir)
    except Exception:
        logging.exception("Failed to save training plots")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()