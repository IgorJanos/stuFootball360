#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
import yaml
import torch

from backstage.trainer import Trainer


BASE_CONFIG_PATH="config"

def train():
    '''
        Select config YAMLs to train
    '''
    configs = [
        "seta-densenet161.yaml"
    ]
    for c in configs:
        trainSingleRun(os.path.join(BASE_CONFIG_PATH, c))



def trainSingleRun(configFile):
    with open(configFile, "r") as f:
        config = yaml.safe_load(f)

    # Load and execute training
    trainer = Trainer(config)
    trainer.optimizer()


if (__name__ == "__main__"):
    train()




