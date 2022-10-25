#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
import torch

from .loggers import Loggers
from .dataset import FootballDataset
from .model import createModel
from .utils import getTqdm, Statistics, k2FromK1
from .losses import MDLD

from torch.utils.data import DataLoader
from torchvision import transforms

#------------------------------------------------------------------------------
#   DataSource class
#------------------------------------------------------------------------------

class DataSource:
    def __init__(self, conf, subset, batchSize):
        self.ds = FootballDataset(conf["folder"], conf[subset], asTensor=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.loader = DataLoader(
            self.ds,
            batch_size=batchSize,
            shuffle=True,
            num_workers=conf["numWorkers"],
            pin_memory=True
        )
        # Iterable loader
        self.itLoader = None

    def get(self):
        if (self.itLoader is None):
            self.itLoader = iter(self.loader)

        # Loop indefinitely
        try:
            sample = next(self.itLoader)
        except (OSError, StopIteration):
            self.itLoader = iter(self.loader)
            sample = next(self.itLoader)

        return sample


#------------------------------------------------------------------------------
#   
#   Trainer class
#
#------------------------------------------------------------------------------

class Trainer:
    def __init__(self, conf):

        # CUDA / CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conf = conf
        self.outputFolder = conf["outputFolder"]       
        self.batchSize = conf["batchSize"]

        # Training params
        self.epochs = conf["epochs"]
        self.itPerEpoch = conf["itPerEpoch"]

        # Data sources
        self.shape = (224, 224)
        self.dsTrain = DataSource(conf["data"], "trainSet", self.batchSize)
        self.dsVal = DataSource(conf["data"], "valSet", self.batchSize)

        # Model
        self.model = createModel(conf["model"])
        self.model = self.model.to(self.device)

        # Optimizer
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=conf["lr"],
            betas=conf["betas"]
        )     
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.opt, gamma=conf["lrDecay"]
        )

        self.log = Loggers()
        self.stats = Statistics(['lossT', 'lossV', 'mdldT', 'mdldV'])

        # Loss functions
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionMDLD = MDLD(shape=(640, 360), device=self.device)


    def setup(self, loggers):
        os.makedirs(self.outputFolder, exist_ok=True)
        self.log = Loggers(loggers)


    def optimize(self):

        # Data paralell !
        model = torch.nn.DataParallel(self.model)

        self.log.trainingStart()
        for i in range(self.epochs):

            self.stats.reset()
            self.log.epochStart(i, self.stats)

            # Train & Validate
            self.trainingPass(i, model)
            self.validationPass(i, model, self.dsVal.loader)

            self.log.epochEnd(i, self.stats)

            self.scheduler.step()

        self.log.trainingEnd()
        

    def trainingPass(self, epoch, model):
        model.train()

        progress = getTqdm(range(self.itPerEpoch))
        progress.set_description("Train {}".format(epoch+1))
        for _ in progress:

            # Get next sample
            x, k = self.dsTrain.get()
            x = x.to(self.device)
            k = k.to(self.device)
            x = self.dsTrain.transform(x)

            # Optimize one step
            self.opt.zero_grad()
            k_hat = model(x)
            loss = self.criterionL2(k_hat, k[:,0:1])
            mdld = self.criterionMDLD(k2FromK1(k_hat), k).mean()
            loss.backward()
            self.opt.step()

            # Update stats
            self.stats.step("lossT", loss.cpu().detach().item())
            self.stats.step("mdldT", mdld.cpu().detach().item())

            # Update progress info
            progress.set_postfix(self.stats.getAvg(["lossT", "mdldT"]))



    def validationPass(self, epoch, model, data):
        model.eval()
        with torch.no_grad():            
            progress = getTqdm(data)
            progress.set_description("Val   {}".format(epoch+1))
            for (x, k) in progress:
                x = x.to(self.device)
                k = k.to(self.device)
                x = self.dsTrain.transform(x)

                # Compute the prediction
                k_hat = model(x)

                # Compute metrics
                loss = self.criterionL2(k_hat, k[:,0:1])
                mdld = self.criterionMDLD(k2FromK1(k_hat), k).mean()

                # Update stats
                self.stats.step("lossV", loss.cpu().detach().item())
                self.stats.step("mdldV", mdld.cpu().detach().item())

                # Update progress info
                progress.set_postfix(self.stats.getAvg(["lossV", "mdldV"]))


