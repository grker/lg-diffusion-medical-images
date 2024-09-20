import pytorch_lightning as pl

from src.utils.hydra_config import ResNetConfig
from building_blocks import *



class ResUNet(pl.LightningModule):
    def __init__(self, config: ResNetConfig):
        super(ResUNet, self).__init__()
        self.num_res_layers = len(config.layers) - 2  # last entry is used for the bridge, first entry is used for the start block
        assert(self.num_res_layers > 0) # at least one res block is needed

        #encoder
        self.startBlock = StartBlock(config.starting_channels, config.layers[0])
        self.resBlocks = nn.ModuleList([ResBlock(config.layers[i], config.layers[i+1]) for i in range(self.num_res_layers)])
        #bridge
        self.bridge = ResBlock(config.layers[-2], config.layers[-1])
        #decoder
        self.upsampleBlocks = nn.ModuleList([DecodeBlock(config.layers[i], config.layers[i-1]) for i in range(self.num_res_layers+1, 0, -1)])
        #output
        self.outputConv = nn.Conv2d(config.layers[1], 1, 1, padding='same')



    def forward(self, x):
        skip_connections = []
        start_block = self.startBlock(x)
        skip_connections.append(start_block)
        
        for l, resBlock in enumerate(self.resBlocks):
            tmp = resBlock(skip_connections[l])
            skip_connections.append(tmp)
        
        tmp = self.bridge(skip_connections[-1])

        for l, upsampleBlock in enumerate(self.upsampleBlocks):
            tmp = upsampleBlock(tmp, skip_connections[-1-l])

        return nn.functional.sigmoid(self.outputConv(tmp))