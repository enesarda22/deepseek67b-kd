import os
from functools import partial
from typing import Optional, Union

import numpy as np
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW, Optimizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.loggers import WandbLogger
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class DistillationModule(L.LightningModule):
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
        self.student_model_name = student_model_name
        self.student_model = None

    def configure_model(self):
        if self.student_model is not None:
            return
        self.student_model = AutoModelForCausalLM.from_pretrained(self.student_model_name, torch_dtype="bfloat16")
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
        self.log("val_loss", outputs.loss, on_epoch=True, prog_bar=True, sync_dist=True)
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

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
        norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), gradient_clip_val)
        self.log("grad_norm", norm, on_step=True, on_epoch=False, prog_bar=True)


if __name__ == "__main__":
    L.seed_everything(42)

    # Paths
    DATASET_PATH = "data/tokenized_finetune_data.npy"
    STUDENT_MODEL = "enesarda22/Llama-3.1-8B-DeepSeek67B-Distilled"
    OUTPUT_DIR = "models/FineTuned-DistLlama-3.1-8B"
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

    # hardware
    PRECISION = "bf16-true"
    ACCELERATOR = "cuda"
    DEVICES = 2

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    STRATEGY = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
    )

    # logger
    wandb_logger = WandbLogger(
        name="Med-FT Llama-3.1-8B-DeepSeek67B-Distilled",
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
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        default_root_dir=OUTPUT_DIR,
        logger=wandb_logger,
        val_check_interval=VAL_CHECK_INTERVAL,
        gradient_clip_val=GRAD_CLIP_VAL,
        accumulate_grad_batches=ACC_GRAD_BATCHES,
        precision=PRECISION,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        strategy=STRATEGY,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(distill_module, train_loader, val_loader)

    # Save final model
    distill_module.student_model.save_pretrained(OUTPUT_DIR)
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    student_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Final model saved to {OUTPUT_DIR}")
