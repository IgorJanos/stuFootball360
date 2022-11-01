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
import numpy as np

from .trainer import DataSource
from .model import createModel
from .utils import getTqdm

class Evaluator:
    def __init__(self, configFile):

        # CUDA / CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the config file
        with open(configFile, "r") as f:
            self.loadConf(yaml.safe_load(f)["train"])



    def loadConf(self, conf):
        self.conf = conf
        self.batchSize = conf["batchSize"]

        # Data sources
        self.shape = (224, 224)
        self.dsVal = DataSource(conf["data"], "valSet", self.batchSize)

        # Model
        self.model = createModel(conf["model"])
        self.model = self.model.to(self.device)

        # Load best model
        bestFileName = os.path.join(conf["outputFolder"], "checkpoint.pt")
        if (os.path.isfile(bestFileName)):
            checkpoint = torch.load(bestFileName, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint["model"])
            self.model = self.model.to(self.device)            
        else:
            print("No best checkpoint.")
            return None


    def evaluate(self):

        dataX = []
        dataY = []

        model = torch.nn.DataParallel(self.model)
        model.eval()
        with torch.no_grad():
            progress = getTqdm(self.dsVal.loader)
            for (x, k) in progress:
                x = x.to(self.device)
                k = k.to(self.device)
                x = self.dsVal.transform(x)

                # Compute the prediction
                k_hat = model(x)
                error_k1 = (k_hat[:,0:1] - k[:,0:1])

                # Store data in the result lists
                dataX.append(k[:,0:1].cpu().detach().numpy())
                dataY.append(error_k1[:,0:1].cpu().detach().numpy())

        # Concatenate the results
        dataX = np.concatenate(dataX, axis=0)
        dataY = np.concatenate(dataY, axis=0)
        return dataX[:,0], dataY[:,0]


