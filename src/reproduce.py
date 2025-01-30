import hydra
import os
import logging
import wandb
import pytorch_lightning as pl
import torch
import sys
import torchvision

from omegaconf import OmegaConf, open_dict
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from wandb import Api

from utils.hydra_config import ReproduceConfig
from models.base_segmentation import create_segmentor


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="reproduce",
)
def main(config: ReproduceConfig):

    logging.getLogger("pytorch_lightning").setLevel(
        logging.INFO
    )  # suppress excessive logs

    # wandb login and config
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    # hydra output dir
    base_path = os.path.dirname(os.getcwd())
    print(f"base_path: {base_path}")
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"log_dir: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    run = wandb.init(
        project="difseg",
        config=wandb.config,
        tags=config.wandb_tags,
        job_type="train",
        dir=log_dir,
    )

    wandb_logger = WandbLogger(log_model=True)

    # save config as artifact
    wandb_run_dir = run.dir
    print(f"wandb_run_dir: {wandb_run_dir}")
    config_yaml = OmegaConf.to_yaml(config)
    with open(os.path.join(wandb_run_dir, "config.yaml"), "w") as f:
        f.write(config_yaml)

    print(f"Is cuda available: {torch.cuda.is_available()}")

    api = Api()

    # load run and its config
    old_run = api.run(f"{config.wandb_username}/{config.wandb_project}/{config.run_id}")
    old_config = OmegaConf.create(old_run.config)

    if torch.cuda.is_available():
        print(f"uses device: {torch.cuda.current_device()}")
    else:
        old_config.trainer.accelerator = "cpu"

    if old_config.seed is None or old_config.seed == -1:
        old_config.seed = pl.seed_everything()
    else:
        pl.seed_everything(old_config.seed)

    old_config.metrics = config.metrics

    print(f"old config metrics: {old_config.metrics}")

    segmentor = create_segmentor(old_config)
    seg_model, train_loader, val_loader, test_loader = segmentor.initialize()

    model_checkpoint = ModelCheckpoint(
        monitor=old_config.trainer.argmax_metric,
        filename="model_{epoch}",
        mode=old_config.trainer.argmax_mode,
        save_top_k=1,
        save_last=True,
        dirpath=log_dir,
    )

    print(f"Initialization done.")

    if old_config.train:
        trainer = pl.Trainer(
            max_epochs=old_config.trainer.max_epochs,
            enable_progress_bar=True,
            callbacks=[model_checkpoint],
            check_val_every_n_epoch=old_config.validation_period,
            log_every_n_steps=1,
            enable_checkpointing=True,
            benchmark=True,
            default_root_dir=log_dir,
            gradient_clip_val=1.0,
            logger=wandb_logger,
            accelerator=old_config.trainer.accelerator,
            devices=1,
            num_sanity_val_steps=0,
            val_check_interval=1.0,
        )

        trainer.fit(seg_model, train_loader, val_loader)

        best_model_path = model_checkpoint.best_model_path
        print(f"Best model path: {best_model_path}")
        trainer.test(seg_model, test_loader, ckpt_path="best", verbose=False)


if __name__ == "__main__":
    main()
