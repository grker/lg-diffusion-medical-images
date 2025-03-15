import copy
import logging
import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from models.base_segmentation import create_segmentor
from utils.helper import create_wandb_tags
from utils.hydra_config import SegmentationConfig


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="segment",
)
def main(config: SegmentationConfig):
    logging.getLogger("pytorch_lightning").setLevel(
        logging.INFO
    )  # suppress excessive logs

    if config.seed is None or config.seed == -1:
        config.seed = pl.seed_everything()
    else:
        pl.seed_everything(config.seed)

    # wandb login and config
    wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb.config = copy.deepcopy(
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )

    # wandb tags
    # if config.wandb_tags is None:
    #     wandb_tags = [
    #         config.project_name,
    #         config.dataset.name,
    #         str(config.dataset.data_path).split("/")[-1],
    #         config.model.name,
    #         "multiclass" if config.dataset.mask_transformer.multiclass else "binary",
    #     ]
    # else:
    #     wandb_tags = config.wandb_tags

    wandb_tags = create_wandb_tags(config)
    print(f"wandb_tags: {wandb_tags}")

    # hydra output dir
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(log_dir, exist_ok=True)

    run = wandb.init(
        project="difseg",
        config=wandb.config,
        tags=wandb_tags,
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
    print(f"uses device: {torch.cuda.current_device()}")

    segmentor = create_segmentor(config)
    seg_model, train_loader, val_loader, test_loader = segmentor.initialize()

    model_checkpoint = ModelCheckpoint(
        monitor=config.trainer.argmax_metric,
        filename="model_{epoch}",
        mode=config.trainer.argmax_mode,
        save_top_k=1,
        save_last=True,
        dirpath=log_dir,
    )

    print("Initialization done.")

    if config.train:
        trainer = pl.Trainer(
            max_epochs=config.trainer.max_epochs,
            enable_progress_bar=True,
            callbacks=[model_checkpoint],
            check_val_every_n_epoch=config.validation_period,
            log_every_n_steps=1,
            enable_checkpointing=True,
            benchmark=True,
            default_root_dir=log_dir,
            gradient_clip_val=1.0,
            logger=wandb_logger,
            accelerator=config.trainer.accelerator,
            devices=1,
            num_sanity_val_steps=1,
            val_check_interval=1.0,
        )

        trainer.fit(seg_model, train_loader, val_loader)

        best_model_path = model_checkpoint.best_model_path
        print(f"Best model path: {best_model_path}")
        trainer.test(seg_model, test_loader, ckpt_path="best", verbose=False)


if __name__ == "__main__":
    main()
