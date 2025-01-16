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
    config_name="ensemble_test",
)
def main(config: TestConfig):

    logging.getLogger("pytorch_lightning").setLevel(
        logging.INFO
    )  # suppress excessive logs

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
    # print(f"uses device: {torch.cuda.current_device()}")

    # initialize wandb api
    api = Api()

    # load run and its config
    old_run = api.run(f"{config.wandb_username}/{config.wandb_project}/{config.run_id}")
    old_config = OmegaConf.create(old_run.config)

    if torch.cuda.is_available():
        print(f"uses device: {torch.cuda.current_device()}")
    else:
        old_config.trainer.accelerator = "cpu"

    # initialize segmentor on test mode
    segmentor = create_segmentor(old_config)
    seg_model_class, test_loader, model_args = segmentor.initialize(test=True)

    # load best model of the run
    model_artifact = wandb.use_artifact(f"model-{config.run_id}:best", type="model")
    model_artifact_dir = model_artifact.download()
    checkpoint_path = os.path.join(model_artifact_dir, "model.ckpt")
    seg_model = seg_model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path, **model_args
    )

    trainer = pl.Trainer(
        enable_progress_bar=True,
        log_every_n_steps=1,
        enable_checkpointing=True,
        benchmark=True,
        default_root_dir=log_dir,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        accelerator=old_config.trainer.accelerator,
        devices=1,
    )

    if hasattr(seg_model, "repetitions_test") and config.repetitions is not None:
        for reps in config.repetitions:
            seg_model.repetitions_test = reps
            trainer.test(seg_model, test_loader)
    else:
        trainer.test(seg_model, test_loader)


if __name__ == "__main__":
    main()
