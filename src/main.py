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

from utils.hydra_config import SegmentationConfig
from models.base_segmentation import create_segmentor


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train_test",
)
def main(config: SegmentationConfig):
    # setup
    logging.getLogger("pytorch_lightning").setLevel(
        logging.INFO
    )  # suppress excessive logs

    if config.seed == -1:
        config.seed = pl.seed_everything()
    else:
        pl.seed_everything(config.seed)

    base_path = os.path.dirname(os.getcwd())
    wandb_path = os.path.join(base_path, "wandb")
    wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb_tags = [
        config.project_name,
        config.dataset.name,
        "multiclass" if config.dataset.mask_transformer.multiclass else "binary"
    ]
    
    wandb.init(
        project="difseg", config=wandb.config, tags=wandb_tags, job_type="train", dir=wandb_path
    )
    wandb_logger = WandbLogger(log_model=True)

    logdir = os.path.join(
        base_path, "lightning_logs", config.project_name, wandb.run.id
    )
    print(f"logdir: {logdir}")
    os.makedirs(logdir, exist_ok=True)
    os.system(f"rm -r {logdir}/*")

    print(f"Is cuda available: {torch.cuda.is_available()}")
    print(f"uses device: {torch.cuda.current_device()}")
    
    segmentor = create_segmentor(config)
    seg_model, train_loader, val_loader, test_loader = segmentor.initialize()
    
    print(f"Initialization done.")

    # images = torch.randn(8, 3, 200, 200)
    # images = torchvision.utils.make_grid(images, nrow=4, padding=10)

    # wandb.log({"pics": wandb.Image(
    #     images,
    #     caption="test"
    # )})

    if config.train:
        trainer = pl.Trainer(
            max_epochs=config.trainer.max_epochs,
            enable_progress_bar=True,
            callbacks=[],
            check_val_every_n_epoch=2,
            log_every_n_steps=1,
            enable_checkpointing=True,
            benchmark=True,
            default_root_dir=logdir,
            gradient_clip_val=1.0,
            logger=wandb_logger,
            accelerator=config.trainer.accelerator,
            devices=1,
            num_sanity_val_steps=0,
            val_check_interval=1.0
        )

        trainer.fit(seg_model, train_loader, val_loader)
        trainer.test(seg_model, test_loader)



if __name__ == "__main__":
    main()
