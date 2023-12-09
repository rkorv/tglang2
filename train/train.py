import os
import math
import pickle

import pytorch_optimizer
import pytorch_lightning as pl
import torchmetrics
from pytorch_metric_learning import losses

import torch
from torch.utils.data import DataLoader

from utils import helper, lang_enum
from utils.dataset import CodeDataset
from model import tglang


DF_PATHONLY_CACHE_PATH = "../../datasets/cache/path_only.pickle"
TEST_SOURCES = ["rosetta", "llama_tasks", "tgdataset", "tgdataset2", "cpp_test"]
MODEL_RESUME = None
LOGS_DIR = "./logs"
BATCH_SIZE = 1024
ACCUMULATE_MAP = {
    60: math.ceil(2048 / BATCH_SIZE),
    90: math.ceil(4096 / BATCH_SIZE),
}
NUM_WORKERS = min(28, BATCH_SIZE)
MAX_EPOCHS = 130

WARMUP_STEPS = 4000
SCHEDULER_STEPS = 200000
LINEAR_STEPS_BEGIN = SCHEDULER_STEPS
LINEAR_STEPS = 10000

# For large
MAX_CODE_LEN = None
TRAIN_SEQ_LEN = 4096
CROP_AUGS = (1, 60)
MODEL_SIZE = "l"

SEED = 9
WEIGHT_DECAY = 1e-1
BETAS = (0.9, 0.95)
LR = 2e-3
MIN_LR = 2e-5
LABEL_SMOOTHING = 0.1

pl.seed_everything(SEED)
torch.set_float32_matmul_precision("medium")


class LanguageClassifier(pl.LightningModule):
    def __init__(self, model_size=MODEL_SIZE, weights=None):
        super().__init__()

        self.model = tglang.get_model(size=model_size)

        self.weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else None

        self.ce_loss = torch.nn.BCEWithLogitsLoss(weight=self.weights, reduction="none")
        self.other_ce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.accuracy = torch.nn.ModuleDict(
            {
                source: torchmetrics.Accuracy(task="multiclass", num_classes=len(lang_enum.languages))
                for source in TEST_SOURCES
            }
        )

    def calc_additional_loss(self, additional_logits, labels):
        cpp_logits, js_logits = additional_logits
        cpp_mask = (labels[:, lang_enum.languages2num["TGLANG_LANGUAGE_CPLUSPLUS"]] > 0) | (
            labels[:, lang_enum.languages2num["TGLANG_LANGUAGE_C"]] > 0
        )
        js_mask = (labels[:, lang_enum.languages2num["TGLANG_LANGUAGE_JAVASCRIPT"]] > 0) | (
            labels[:, lang_enum.languages2num["TGLANG_LANGUAGE_TYPESCRIPT"]] > 0
        )

        if cpp_mask.any():
            cpp_logits = cpp_logits[cpp_mask]
            cpp_labels = labels[cpp_mask]
            gt = torch.zeros((len(cpp_labels), 2), dtype=torch.float32).to(labels.device)
            gt[:, 0] = cpp_labels[:, lang_enum.languages2num["TGLANG_LANGUAGE_CPLUSPLUS"]]
            gt[:, 1] = cpp_labels[:, lang_enum.languages2num["TGLANG_LANGUAGE_C"]]

            cpp_loss = self.other_ce_loss(cpp_logits, gt)
            cpp_loss = cpp_loss.mean()
        else:
            cpp_loss = 0

        if js_mask.any():
            js_logits = js_logits[js_mask]
            js_labels = labels[js_mask]
            gt = torch.zeros((len(js_labels), 2), dtype=torch.float32).to(labels.device)
            gt[:, 0] = js_labels[:, lang_enum.languages2num["TGLANG_LANGUAGE_JAVASCRIPT"]]
            gt[:, 1] = js_labels[:, lang_enum.languages2num["TGLANG_LANGUAGE_TYPESCRIPT"]]

            js_loss = self.other_ce_loss(js_logits, gt)
            js_loss = js_loss.mean()
        else:
            js_loss = 0

        return cpp_loss, js_loss

    def calc_loss(self, code_logits, other_logits, labels, features, additional_logits):
        num_labels = len(lang_enum.languages)
        # Code
        ce_code_loss = self.ce_loss(code_logits, labels)
        ce_code_loss = (ce_code_loss).mean() * num_labels

        # Other
        gt = torch.zeros((len(labels), 2), dtype=torch.float32).to(labels.device)
        gt[:, 0] = labels[:, 0]
        gt[:, 1] = 1 - gt[:, 0]

        ce_other_loss = self.other_ce_loss(other_logits, gt)
        ce_other_loss = (ce_other_loss).mean()

        # Additional
        cpp_loss, js_loss = self.calc_additional_loss(additional_logits, labels)

        # Total
        loss = ce_code_loss + ce_other_loss * 0.5 + cpp_loss * 0.2 + js_loss * 0.3
        loss_stat = {
            "ce_code_loss": ce_code_loss,
            "ce_other_loss": ce_other_loss,
            "cpp_loss": cpp_loss,
            "js_loss": js_loss,
        }

        return loss, loss_stat

    def forward(self, input_ids, naming_types, group_types, line_ids, positions, attention_masks):
        return self.model(input_ids, naming_types, group_types, line_ids, positions, attention_masks)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (input_ids, naming_types, group_types, line_ids, positions, labels, attention_masks) = batch
        max_label = 1 - LABEL_SMOOTHING
        labels[labels > max_label] = max_label
        code_logits, other_logits, features, additional_logits = self(
            input_ids, naming_types, group_types, line_ids, positions, attention_masks
        )
        loss, _ = self.calc_loss(code_logits, other_logits, labels, features, additional_logits)

        probs = torch.softmax(code_logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds[torch.softmax(other_logits, dim=1)[:, 0] > 0.5] = 0
        preds[probs.max(dim=1).values < 0.3] = 0

        if (TEST_SOURCES[dataloader_idx] == "tgdataset") or (TEST_SOURCES[dataloader_idx] == "tgdataset2"):
            preds[preds > 0] = 1

        self.accuracy[TEST_SOURCES[dataloader_idx]](preds, labels.argmax(dim=1))

        self.log(f"val/loss", loss, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        for source, accuracy in self.accuracy.items():
            acc_name = f"val/acc/{source}"
            self.log(
                acc_name,
                accuracy.compute(),
                add_dataloader_idx=False,
                on_epoch=True,
                sync_dist=True,
            )
            accuracy.reset()

    def training_step(self, batch, batch_idx):
        input_ids, naming_types, group_types, line_ids, positions, labels, attention_masks = batch
        max_label = 1 - LABEL_SMOOTHING
        labels[labels > max_label] = max_label

        code_logits, other_logits, features, additional_logits = self(
            input_ids, naming_types, group_types, line_ids, positions, attention_masks
        )
        loss, loss_stat = self.calc_loss(code_logits, other_logits, labels, features, additional_logits)

        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        for name, value in loss_stat.items():
            self.log(f"train/loss_{name}", value, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        parameters = self.model.parameters()
        optimizer = pytorch_optimizer.AdaBelief(parameters, LR, BETAS, WEIGHT_DECAY)
        train_scheduler = helper.InverseSquareRootLRScheduler(
            optimizer,
            LR,
            MIN_LR,
            SCHEDULER_STEPS,
            LINEAR_STEPS_BEGIN,
            LINEAR_STEPS,
            WARMUP_STEPS,
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": train_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "strict": True,
                },
            },
        )


def split_train_test(df, test_sources):
    test_dfs = [df[df["source"] == source] for source in test_sources]
    train_df = df[~df["source"].isin(test_sources)]
    return train_df, test_dfs


if __name__ == "__main__":
    with open(DF_PATHONLY_CACHE_PATH, "rb") as f:
        df = pickle.load(f)

    if MAX_CODE_LEN is not None:
        df = df[df.code_len <= MAX_CODE_LEN]

    traindf, testdfs = split_train_test(df, TEST_SOURCES)
    train_loader = DataLoader(
        CodeDataset(traindf, TRAIN_SEQ_LEN, crop_augs=CROP_AUGS),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=CodeDataset.collate_fn,
    )
    test_loaders = [
        DataLoader(
            CodeDataset(testdf, TRAIN_SEQ_LEN, aug=False),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            collate_fn=CodeDataset.collate_fn,
        )
        for testdf in testdfs
    ]

    weights = helper.compute_class_weights_sqrt(traindf, 2.0)
    model = LanguageClassifier(weights=weights)

    exp_name = helper.generate_experiment_name({"bs": BATCH_SIZE, "epochs": MAX_EPOCHS}, prefix=MODEL_SIZE)
    exp_dir = os.path.join(LOGS_DIR, exp_name)

    trainer_cfg = {
        "accelerator": "gpu",
        "devices": 2,
        "sync_batchnorm": True,
        "logger": [
            pl.loggers.TensorBoardLogger(save_dir=exp_dir),
            pl.loggers.WandbLogger(name=exp_name, project="tglang"),
        ],
        "precision": "16-mixed",
        "max_epochs": MAX_EPOCHS,
        "log_every_n_steps": 100,
        "gradient_clip_val": 2.0,
        "val_check_interval": 1.0,
        "callbacks": [
            pl.callbacks.ModelCheckpoint(dirpath=exp_dir, every_n_epochs=1, save_top_k=-1),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.GradientAccumulationScheduler(scheduling=ACCUMULATE_MAP),
        ],
    }

    trainer = pl.Trainer(**trainer_cfg)

    _ = trainer.fit(
        model,
        ckpt_path=MODEL_RESUME,
        train_dataloaders=train_loader,
        val_dataloaders=test_loaders,
    )
