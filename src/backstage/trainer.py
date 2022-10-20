#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import torch

from .loggers import Loggers
from .dataset import FootballDataset
from .model import createModel
from .utils import getTqdm, Statistics, k2FromK1
from .losses import MDLD

from torch.utils.data import DataLoader

#------------------------------------------------------------------------------
#   DataSource class
#------------------------------------------------------------------------------

class DataSource:
    def __init__(self, conf, subset, batchSize):
        self.ds = FootballDataset(conf["folder"], conf[subset], asTensor=True)
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
        self.dsTrain = DataSource(conf["data"], "trainSet", self.batchSize)
        self.dsVal = DataSource(conf["data"], "valSet", self.batchSize)

        # Model
        self.model = createModel(conf["model"])

        # Optimizer
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=conf["lr"],
            betas=conf["betas"]
        )     

        self.log = Loggers()
        self.stats = Statistics(['lossT', 'lossV', 'mdldT', 'mdldV'])

        # Loss functions
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionMDLD = MDLD()


    def optimize(self):

        # Data paralell !
        model = torch.nn.DataParallel(self.model)

        self.log.trainingStart()
        for i in range(self.epochs):

            self.stats.reset()
            self.log.epochStart(i, self.stats)

            # Train
            self.trainingPass(i, model)

            # Validate
            self.validationPass(model, self.dsVal.loader)

            self.log.epochEnd(i, self.stats)

        self.log.trainingEnd()
        

    def trainingPass(self, epoch, model):
        model.train()

        progress = getTqdm(range(self.itPerEpoch))
        progress.set_description("Epoch {}".format(epoch+1))
        for _ in progress:

            # Get next sample
            x, k = self.dsTrain.get()
            x = x.to(model.device)
            k = k.to(model.device)

            # Optimize one step
            self.opt.zero_grad()
            k_hat = model(x)
            loss = self.criterionL2(k_hat, k[:,0:1])
            mdld = self.criterionMDLD(k2FromK1(k_hat), k)
            loss.backward()
            self.opt.step()

            # Update stats
            self.stats.step("lossT", loss.cpu().detach().item())
            self.stats.step("mdldT", mdld.cpu().detach().item())

            # Update progress info
            progress.set_postfix(self.stats.getAvg(["lossT", "mdldT"]))



    def validationPass(self, model, data):
        model.eval()
        with torch.no_grad():            
            for (x, k) in data:
                x = x.to(model.device)
                k = k.to(model.device)

                # Compute the prediction
                k_hat = model(x)

                # Compute metrics
                loss = self.criterionL2(k_hat, k[:,0:1])
                mdld = self.criterionMDLD(k2FromK1(k_hat), k)

                # Update stats
                self.stats.step("lossV", loss.cpu().detach().item())
                self.stats.step("mdldV", mdld.cpu().detach().item())



