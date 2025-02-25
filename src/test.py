import hydra
import torch

from utils.hydra_config import LossGuidanceInferenceConfig
from utils.visualize import visualize_gradients


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="loss_guidance",
)
def main(config: LossGuidanceInferenceConfig):
    print("Hello World", flush=True)

    import wandb

    print("wandb imported", flush=True)
    wandb.init(mode="online")

    grads = torch.randn(3, 256, 256) * 4
    original_image = torch.randn(3, 256, 256)

    visualize_gradients(
        grads,
        1,
        0,
        0,
        original_image,
        commit=True,
    )

    print("finished")


if __name__ == "__main__":
    main()
