import os

import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


def print_parameter_counts(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} -- Total parameters: {total_params:,}")


class DistillationModule(pl.LightningModule):
    def __init__(
            self,
            student_model_name,
            learning_rate=1e-4,
            warmup_steps=1000,
            betas=(0.9, 0.95),
            eps = 1e-8,
            weight_decay = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Load models
        self.student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
        print_parameter_counts(self.student_model, student_model_name)
        self.student_model = torch.compile(self.student_model)

    def forward(self, input_ids, attention_mask):
        return self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch[0]

        # forward student
        outputs = self.student_model(
            input_ids=input_ids,
            labels=input_ids,
        )
        lr = self.optimizers().param_groups[0]['lr']

        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss", outputs.loss, on_step=True, on_epoch=False, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch[0]

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
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
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
    DATASET_PATH = "data/tokenized_finetune_data.npy"
    STUDENT_MODEL = "enesarda22/Llama-3.2-1B-DeepSeek67B-Distilled"
    OUTPUT_DIR = "models/FineTuned-DistLlama-3.2-1B"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    EPOCHS = 1
    BATCH_SIZE = 2
    ACC_GRAD_BATCHES = 1
    MAX_LENGTH = 2048
    VAL_CHECK_INTERVAL = 0.25

    # optimizer
    GRAD_CLIP_VAL = 1.0
    LEARNING_RATE = 1e-5
    WARMUP_STEPS = 200
    BETAS = (0.9, 0.95)
    EPS = 1e-8
    WEIGHT_DECAY = 0.1

    wandb_logger = WandbLogger(
        name="Med-FT Llama-3.2-1B-DeepSeek67B-Distilled",
        project="deepseek67b-kd",
    )

    # Load Dataset
    print(f"Loading dataset from {DATASET_PATH} ...")
    input_ids_np = np.load(DATASET_PATH)
    input_ids_np = input_ids_np.reshape(-1, MAX_LENGTH)

    ds = TensorDataset(torch.tensor(input_ids_np, dtype=torch.long))
    train_ds, val_ds = torch.utils.data.random_split(ds, [0.9, 0.1])
    print(f"Train set: {len(train_ds)}, Val set: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
    )

    torch.set_float32_matmul_precision("high")

    # Create Lightning Module (DistillationModule)
    distill_module = DistillationModule(
        student_model_name=STUDENT_MODEL,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
    )

    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath = OUTPUT_DIR,
        filename = "{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_on_train_epoch_end=False,
    )

    # Setup Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        default_root_dir=OUTPUT_DIR,
        logger=wandb_logger,
        val_check_interval=VAL_CHECK_INTERVAL,
        gradient_clip_val=GRAD_CLIP_VAL,
        accumulate_grad_batches=ACC_GRAD_BATCHES,
        precision="bf16-mixed",
        accelerator="auto",
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(distill_module, train_loader, val_loader)

    # Save final model
    distill_module.student_model.save_pretrained(OUTPUT_DIR)
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    student_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Final model saved to {OUTPUT_DIR}")
