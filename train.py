import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


def print_parameter_counts(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} -- Total parameters: {total_params:,}")


def distillation_loss_function(
        student_logits,
        teacher_logits,
        ce_loss,
        alpha=0.5,
        temperature=1.0
):
    """
    Combined cross-entropy (w/ ground truth) + distillation (KL) loss.
    """

    # Flatten logits
    student_logits_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))

    # Apply temperature
    student_logits_temp = student_logits_flat / temperature
    teacher_logits_temp = teacher_logits_flat / temperature

    # Convert teacher logits to probabilities
    teacher_probs = torch.softmax(teacher_logits_temp, dim=-1)
    log_student_probs = torch.log_softmax(student_logits_temp, dim=-1)

    # KL Divergence
    kl_div = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-9) - log_student_probs), dim=-1)
    distill_loss = torch.mean(kl_div) * (temperature ** 2)

    loss = alpha * distill_loss + (1 - alpha) * ce_loss
    return loss


def data_collator(features, tokenizer):
    """
    Convert each feature to Tensors, pad, and set labels for LM.
    """
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]

    # pad
    padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


class DistillationModule(pl.LightningModule):
    def __init__(
            self,
            student_model_name_or_path,
            teacher_model_name_or_path,
            tokenizer,
            alpha=0.8,
            temperature=1.0,
            learning_rate=1e-4,
            warmup_steps=1000,
            total_steps=100_000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])

        self.tokenizer = tokenizer

        self.alpha = alpha
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Load models
        self.student_model = AutoModelForCausalLM.from_pretrained(student_model_name_or_path)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name_or_path)
        print_parameter_counts(self.student_model, STUDENT_CKPT)
        print_parameter_counts(self.teacher_model, TEACHER_CKPT)

        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        return self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Forward student
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            return_dict=True
        )
        student_logits = student_outputs.logits

        # Forward teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                return_dict=True
            )
        teacher_logits = teacher_outputs.logits

        # Distillation loss
        loss = distillation_loss_function(
            student_logits,
            teacher_logits,
            student_outputs.loss,
            alpha=self.alpha,
            temperature=self.temperature
        )

        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        # Standard AdamW
        optimizer = AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
        )

        # Linear scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


if __name__ == "__main__":
    seed_everything(42)

    # Paths
    DATASET_PATH = "data/tokenized_ds"
    STUDENT_CKPT = "distilgpt2"
    TEACHER_CKPT = "gpt2"
    OUTPUT_DIR = "models/lightning_student_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BATCH_SIZE = 2
    EPOCHS = 1
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 1000

    # Load Dataset
    print(f"Loading dataset from {DATASET_PATH} ...")
    ds = load_from_disk(DATASET_PATH)
    ds = ds.shuffle(seed=42)

    # Simple 90/10 split
    train_size = int(0.9 * len(ds))
    train_ds = ds.select(range(train_size))
    val_ds = ds.select(range(train_size, len(ds)))

    print(f"Train set: {len(train_ds)}, Val set: {len(val_ds)}")

    # Build Dataloaders
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_CKPT)


    def train_collate_fn(features):
        return data_collator(features, tokenizer)


    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=train_collate_fn
    )

    # Create Lightning Module (DistillationModule)
    distill_module = DistillationModule(
        student_model_name_or_path=STUDENT_CKPT,
        teacher_model_name_or_path=TEACHER_CKPT,
        tokenizer=tokenizer,
        alpha=0.8,
        temperature=2.0,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        total_steps=(len(train_ds) // BATCH_SIZE) * EPOCHS,
    )

    # Setup Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        default_root_dir=OUTPUT_DIR,
        precision=16 if torch.cuda.is_available() else 32,
        # optional mixed precision
        accelerator="gpu" if torch.cuda.is_available() else "mps",
        devices=1,  # or more GPUs
        val_check_interval=0.25,  # validate 4 times per epoch, for example
    )
    trainer.fit(distill_module, train_loader, val_loader)

    # Save final model
    distill_module.student_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Final model saved to {OUTPUT_DIR}")
