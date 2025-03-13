import os

import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


def print_parameter_counts(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} -- Total parameters: {total_params:,}")


class DistillationModule(pl.LightningModule):
    def __init__(
            self,
            student_model_name,
            alpha=0.8,
            temperature=1.0,
            learning_rate=1e-4,
            warmup_steps=1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.alpha = alpha
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

        # Load models
        self.student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
        print_parameter_counts(self.student_model, STUDENT_MODEL)

    def forward(self, input_ids, attention_mask):
        return self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        # forward student
        outputs = self.student_model(
            input_ids=input_ids,
            labels=input_ids,
        )

        self.log("train_loss", outputs.loss, on_step=True, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        outputs = self.student_model(
            input_ids=input_ids,
            labels=input_ids,
        )
        self.log("val_loss", outputs.loss, on_epoch=True, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


if __name__ == "__main__":
    seed_everything(42)

    # Paths
    DATASET_PATH = "data/tokenized_data.npy"
    STUDENT_MODEL = "meta-llama/Llama-3.2-1B"
    OUTPUT_DIR = "models/DistLlama-3.2-1B"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BATCH_SIZE = 1
    EPOCHS = 10
    MAX_LENGTH = 2048
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 1000
    ALPHA = 0.8
    TEMPERATURE = 2.0

    wandb_logger = WandbLogger(
        name="T: DeepSeek-67B, S: Llama-3.2-1B",
        project="deepseek67b-kd",
    )

    # Load Dataset
    print(f"Loading dataset from {DATASET_PATH} ...")
    input_ids_np = np.load(DATASET_PATH)
    input_ids_np = input_ids_np.reshape(-1, BATCH_SIZE, MAX_LENGTH)

    ds = TensorDataset(torch.tensor(input_ids_np, dtype=torch.long))
    train_ds, val_ds = torch.utils.data.random_split(ds, [0.9, 0.1])
    print(f"Train set: {len(train_ds)}, Val set: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    torch.set_float32_matmul_precision("high")

    # Create Lightning Module (DistillationModule)
    distill_module = DistillationModule(
        student_model_name=STUDENT_MODEL,
        alpha=ALPHA,
        temperature=TEMPERATURE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
    )
    distill_module = torch.compile(distill_module)

    # Setup Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        default_root_dir=OUTPUT_DIR,
        logger=wandb_logger,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        accumulate_grad_batches=32,
        precision="bf16-mixed",
        accelerator="auto",
    )
    trainer.fit(distill_module, train_loader, val_loader)

    # Save final model
    distill_module.student_model.save_pretrained(OUTPUT_DIR)
    print(f"Done! Final model saved to {OUTPUT_DIR}")
