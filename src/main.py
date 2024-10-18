import hydra
import os
import logging
import wandb
import pytorch_lightning as pl
import torch

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

    wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project="difseg", config=wandb.config, tags=config.wandb_tags, job_type="train"
    )
    wandb_logger = WandbLogger(log_model=True)

    logdir = os.path.join(
        os.getcwd(), "lightning_logs", config.project_name, wandb.run.id
    )
    os.makedirs(logdir, exist_ok=True)
    os.system(f"rm -r {logdir}/*")

    print(f"Is cuda available: {torch.cuda.is_available()}")
    print(f"uses device: {torch.cuda.current_device()}")
    

    segmentor = create_segmentor(config)
    seg_model, train_loader, val_loader, test_loader = segmentor.initialize()
    # dataset, dataloader = segmentor.create_dataset_dataloader(config.dataset, config.dataloader)
    # train_loader, val_loader, test_loader = dataloader
    # image_height, image_width = dataset.get_image_height(), dataset.get_image_width()

    # with open_dict(config.model):
    #     config.model.image_size = (image_height, image_width)
    #     config.diffusion.img_size = (image_height, image_width)

    # model = segmentor.create_model(config.model)
    # diffusion = segmentor.create_diffusion(config.diffusion, model)

    # segmentation_model = segmentor.create_segmentation_model(config)
    
    
    print(f"Initialization done.")
    if config.train:
        trainer = pl.Trainer(
            max_epochs=config.trainer.max_epochs,
            enable_progress_bar=True,
            callbacks=[],
            check_val_every_n_epoch=1,
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



if __name__ == "__main__":
    main()
