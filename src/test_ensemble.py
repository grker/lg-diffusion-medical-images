import hydra
import os
import logging
import wandb
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from wandb import Api
from omegaconf import OmegaConf

from utils.hydra_config import TestConfig, SegmentationConfig
from models.base_segmentation import create_segmentor

@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="test_ensemble",
)
def main(config: TestConfig):

    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)  # suppress excessive logs

    if config.seed is None or config.seed == -1:
        config.seed = pl.seed_everything()
    else:
        pl.seed_everything(config.seed)

    # get hydra output dir
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"log_dir: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project="difseg",
        config=wandb.config,
        tags=config.wandb_tags,
        job_type="test",
        dir=log_dir,
    )

    wandb_logger = WandbLogger(log_model=True)

    wandb_run_dir = run.dir
    # save config
    config_yaml = OmegaConf.to_yaml(config)
    with open(os.path.join(wandb_run_dir, "config.yaml"), "w") as f:
        f.write(config_yaml)

    print(f"Is cuda available: {torch.cuda.is_available()}")
    print(f"uses device: {torch.cuda.current_device()}")

    # initialize wandb api
    api = Api()

    # load run to test on
    run = api.run(f"{config.wandb_username}/{config.wandb_project}/{config.run_id}")
    config_file = run.file("config.yaml")
    config_file.download(replace=True)
    run.download(log_dir, replace=True)

    with open(os.path.join(log_dir, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)

    model_config = OmegaConf.create(config)
    if isinstance(model_config, SegmentationConfig):
        print("is segmentation config")
    else:
        print("is not segmentation config")

    segmentor = create_segmentor(model_config)
    seg_model, train_loader, val_loader, test_loader = segmentor.initialize()

    trainer = pl.Trainer(
            max_epochs=config.trainer.max_epochs,
            enable_progress_bar=True,
            callbacks=[],
            check_val_every_n_epoch=config.validation_period,
            log_every_n_steps=1,
            enable_checkpointing=True,
            benchmark=True,
            default_root_dir=log_dir,
            gradient_clip_val=1.0,
            logger=wandb_logger,
            accelerator=config.trainer.accelerator,
            devices=1,
            num_sanity_val_steps=0,
            val_check_interval=1.0
        )
    
    # TODO: load best model of the run
        
    trainer.test(seg_model, test_loader)
