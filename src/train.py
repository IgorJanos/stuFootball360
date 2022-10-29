#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
from re import S
import yaml
import torch

from backstage.trainer import Trainer
from backstage.loggers import CsvLogger, ModelCheckpoint, ResultSampler, PSNR_SSIM_Sampler

BASE_CONFIG_PATH="config"

def train():
    '''
        Select config YAMLs to train
    '''
    configs = [
        #"seta-densenet161.yaml",
        #"setb-densenet161.yaml",
        #"setc-densenet161.yaml",
        "seta-resnet152.yaml",
        "setb-resnet152.yaml",
        "setc-resnet152.yaml"
        #"seta-efficientnetb5.yaml",
        #"setb-efficientnetb5.yaml",
        #"setc-efficientnetb5.yaml"
    ]
    for c in configs:
        trainSingleRun(os.path.join(BASE_CONFIG_PATH, c))



def trainSingleRun(configFile):

    print("")
    print("Training: ", configFile)

    with open(configFile, "r") as f:
        config = yaml.safe_load(f)

    # Load and execute training
    trainer = Trainer(config["train"])
    trainer.setup([
        CsvLogger(trainer, "training.csv"),
        ModelCheckpoint(trainer, singleFile=True, bestName="mdldV"),
        ResultSampler(
            trainer, 
            scaleShape=(640, 360), 
            indices=config["train"]["evalIndices"]
        ),
        PSNR_SSIM_Sampler(trainer)
    ])
    trainer.optimize()


if (__name__ == "__main__"):
    train()




