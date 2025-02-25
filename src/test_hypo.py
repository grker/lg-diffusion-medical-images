import hydra
import torch

from models.base_segmentation import create_segmentor
from utils.hydra_config import SegmentationConfig


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train_test",
)
def main(config: SegmentationConfig):
    segmentor = create_segmentor(config)
    dataset = segmentor.create_dataset(config.dataset)
    seg_model, train_loader, val_loader, test_loader = segmentor.initialize()

    seg_model.noise_tester(dataset, 256)
    # from dataset.mnist import MNISTDataset, M2NISTDataset
    # dataset = M2NISTDataset(config.dataset)

    # from models.dmiise.diffusion import Diffusion
    # diffusion = Diffusion(config.diffusion)

    # image, mask = dataset[56]
    # print(f"Image: {type(image)}")
    # print(f"type mask: {type(mask)}")
    # visualize_sampling_res(image, mask, mask, saving=False)

    # steps = 10
    # for t in range(0, 500, steps):
    #     timestep = torch.ones(1, dtype=torch.int64) * t

    #     noisy_image, _ = diffusion.q_samples(image, timestep)
    #     print(f"type noisy_image: {type(noisy_image)}")
    #     visualize_sampling_res(image, noisy_image, mask, saving=False)

    t = torch.randn((3, 3), requires_grad=False)
    t = t.requires_grad_()


if __name__ == "__main__":
    main()
