import copy
import logging
import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

import wandb
from models.base_segmentation import create_segmentor
from utils.helper import create_wandb_tags
from utils.hydra_config import LossGuidanceInferenceConfig
from wandb import Api


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="loss_guidance",
)
def main(config: LossGuidanceInferenceConfig):
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

    wandb.config = copy.deepcopy(
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )
    print(f"config: {config}")

    run = wandb.init(
        project="difseg",
        config=wandb.config,
        tags=None,
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

    print(f"using run id: {config.run_id}")
    # load run and its config
    old_run = api.run(f"{config.wandb_username}/{config.wandb_project}/{config.run_id}")
    old_config = OmegaConf.create(old_run.config)

    if old_config.project_name == "unet_seg":
        raise ValueError(
            "Loss guidance is not supported for UNet segmentation. Please specify a trained model with the project name 'dmiise'."
        )

    # add the loss guidance config to the diffusion config
    print(f"new config: {config}")
    print(f"old config: {old_config}")
    old_config.diffusion.loss_guidance = config.loss_guidance
    old_config.loss_guidance = config.loss_guidance
    old_config.dataloader.val_batch_size = config.test_batch_size

    if torch.cuda.is_available():
        print(f"cuda available, using device: {torch.cuda.current_device()}")
    else:
        print("cuda not available, using cpu")
        old_config.trainer.accelerator = "cpu"

    if config.metrics is not None:
        old_config.metrics = config.metrics

    wandb_tags = create_wandb_tags(old_config)
    wandb_tags.append(config.run_id)
    print(f"wandb_tags: {wandb_tags}")
    run.tags = wandb_tags

    segmentor = create_segmentor(old_config, loss_guided=True)
    seg_model_class, test_loader, model_args = segmentor.initialize(test=True)

    print(f"number of batches in test loader: {len(test_loader)}")
    print(f"number of images in test loader: {len(test_loader.dataset)}")

    # load best model of the run
    model_artifact = wandb.use_artifact(f"model-{config.run_id}:best", type="model")
    model_artifact_dir = model_artifact.download()
    checkpoint_path = os.path.join(model_artifact_dir, "model.ckpt")
    seg_model = seg_model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path, **model_args
    )

    print
    trainer = pl.Trainer(
        max_epochs=old_config.trainer.max_epochs,
        enable_progress_bar=True,
        log_every_n_steps=1,
        enable_checkpointing=True,
        benchmark=True,
        default_root_dir=log_dir,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        accelerator=old_config.trainer.accelerator,
        devices=1,
        num_sanity_val_steps=0,
        inference_mode=False,
    )
    print("start with the testing")
    for rep in config.repetitions:
        print(f"testing with repetition {rep}")
        seg_model.repetitions_test = rep
        trainer.test(seg_model, test_loader)


if __name__ == "__main__":
    main()
