This is the repository to the master thesis "Loss-guided Diffusion In Medical Image Segmentation" by Stefan Kramer. 

### Requirements
After cloning the repository, one has to initialize the underlying submodule https://github.com/grker/Betti-matching by running the command
```git submodule update --init --recursive```
inside the repo. 

To set up the python environment, a the ```requirements.txt``` file is provided in the repository. If you'd like to run the experiments on a GPU, please use the CUDA-enabled build. For example: 
```
conda install pytorch=2.3.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Finally, for the persistence homology guidance some code of the repository https://github.com/bruel-gabrielsson/TopologyLayer?tab=readme-ov-file by Gabrielsson et al. was used. It can be directly included into the environment via 
```
pip install git+https://github.com/bruel-gabrielsson/TopologyLayer.git
```
If the direct addition to the environment does not work, you might want to clone it into the project and change ```c++14``` in ```setup.py``` to ```c++17``` and run ```python setup.py install```. If neither of the approaches work, we refer to their instruction in their ```README.md``` file.


### Experiments
There two files which can be run. Here, we'll give a brief explaination about what their purpose is. All experiments use pytorch lightning and run over W&B. We encourage you to set up a W&B account for this. 

- ```main.py```: trains a diffusion model, saves the weights of the best model on its W&B run
- ```loss_guidance.py```: loads the weights of a trained diffusion model and applies the loss guided generation process

The file ```test_ensemble.py``` is not adapted to the newest settings and may be not executable anymore. 

#### Config
In the folder ```conf```, you'll find all possible configuration parameters. The base files are ```segment.yaml``` for the training and ```loss_guidance.yaml``` for the loss-guided inference part. There are already multiple files with possilbe options for the subcategories. One can either add new files with different values, change the values of the existing files or pass the values directly with the command. We use hydra for the configuration, so for example the parameter ```batch_size``` which is part of the subcategory ```dataloader``` can be overwritten when running ```main.py``` by adding
```
dataloader.batch_size=64
```
to the command line. 


### Datasets

#### BCCD
https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask

#### ACDC
https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

#### Multi Digit MNIST
https://www.kaggle.com/datasets/farhanhubble/multimnistm2nist/code



### Citation

The code of the submodule was mainly written by N. Stucki and S. Shit. If you use this work in your research, please cite their work with the following. 
```
@misc{stucki2022topologicallyfaithfulimagesegmentation,
      title={Topologically faithful image segmentation via induced matching of persistence barcodes}, 
      author={Nico Stucki and Johannes C. Paetzold and Suprosanna Shit and Bjoern Menze and Ulrich Bauer},
      year={2022},
      eprint={2211.15272},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2211.15272}, 
}
```

Further, we integrated the code of Gabrielsson's work which should be cited as well. 

```
@misc{brüelgabrielsson2020topology,
      title={A Topology Layer for Machine Learning}, 
      author={Rickard Brüel-Gabrielsson and Bradley J. Nelson and Anjan Dwaraknath and Primoz Skraba and Leonidas J. Guibas and Gunnar Carlsson},
      year={2020},
      eprint={1905.12200},
}
```