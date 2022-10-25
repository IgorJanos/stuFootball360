#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
import wandb
import torch
import cv2

from .utils import rescale, undistort, toNumpyImage, getTqdm, k2FromK1
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from .losses import MDLD

#------------------------------------------------------------------------------
#   Logger baseclass
#------------------------------------------------------------------------------

class Logger:
    def trainingStart(self):
        pass

    def trainingEnd(self):
        pass

    def epochStart(self, epoch, stats):
        pass

    def epochEnd(self, epoch, stats):
        pass

#------------------------------------------------------------------------------
#   Loggers baseclass
#------------------------------------------------------------------------------

class Loggers(Logger):
    def __init__(self, loggers=[]):
        self.loggers = loggers

    def trainingStart(self):
        for l in self.loggers:
            l.trainingStart()

    def trainingEnd(self):
        for l in self.loggers:
            l.trainingEnd()

    def epochStart(self, epoch, stats):
        for l in self.loggers:
            l.epochStart(epoch, stats)

    def epochEnd(self, epoch, stats):
        for l in self.loggers:
            l.epochEnd(epoch, stats)


#------------------------------------------------------------------------------
#   CsvLogger baseclass
#------------------------------------------------------------------------------

class CsvLogger(Logger):
    def __init__(self, trainer, filename):
        self.trainer = trainer
        self.filename = os.path.join(self.trainer.outputFolder, filename)
        self.file = None
        self.lines = 0
        self.separator = ','

    def trainingStart(self):
        self.lines = 0
        self.file = open(self.filename, "w")

    def trainingEnd(self):
        if (self.file is not None):
            self.file.close()
            self.file = None

    def epochEnd(self, epoch, stats):
        if (self.file is not None):
            line = ""
            s = stats.getAvg()

            if (self.lines == 0):
                # CSV header
                line = self.separator.join(["epoch", "lr"] + list(s.keys()))
                self.file.write(line + "\n")

            # Join all values
            values = [ "{}".format(s[k]) for k in s.keys() ]
            values = [ 
                "{}".format(epoch+1),
                "{}".format(self.trainer.opt.param_groups[0]['lr'])
            ] + values
            line = self.separator.join(values)
            self.file.write(line + "\n")
            self.file.flush()
            self.lines += 1


#------------------------------------------------------------------------------
#   WandBLogger 
#------------------------------------------------------------------------------

class WandBLogger(Logger):
    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.run = None

    def trainingStart(self):
        wandb.config = self.config
        self.run = wandb.init(config=self.config, **self.kwargs)

    def trainingEnd(self):
        if (self.run is not None):
            wandb.finish()
            self.run = None

    def epochEnd(self, epoch, stats):
        s = stats.getAvg()
        self.run.log(s)



#------------------------------------------------------------------------------
#   ModelCheckpoint 
#------------------------------------------------------------------------------

class ModelCheckpoint(Logger):
    def __init__(self, trainer, singleFile=True):
        self.trainer = trainer
        self.folder = trainer.outputFolder
        self.singleFile = singleFile

    def trainingEnd(self):
        self.saveCheckpoint("model.pt")

    def epochEnd(self, epoch, stats):
        if (self.singleFile):
            self.saveCheckpoint("model.pt")
        else:
            self.saveCheckpoint("checkpoint-{:04d}.pt".format(epoch+1))

    def saveCheckpoint(self, fn):
        checkpoint = {
            "model": self.trainer.model.state_dict(),
            "opt": self.trainer.opt.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.folder, fn))



#------------------------------------------------------------------------------
#   ResultSampler
#------------------------------------------------------------------------------

class ResultSampler(Logger):
    def __init__(self, trainer, scaleShape, indices=None):
        self.trainer = trainer
        self.folder = os.path.join(self.trainer.outputFolder, "samples")
        self.scaleShape = scaleShape
        os.makedirs(self.folder, exist_ok=True)

        # Get a few samples
        lImages = []
        lK = []
        if (indices is not None):
            for i in indices:
                x, k = self.trainer.dsVal.ds[i]
                lImages.append(x.unsqueeze(0))
                lK.append(torch.from_numpy(k).unsqueeze(0))

        # Store the images & labels
        self.images = torch.cat(lImages, dim=0)
        self.k = torch.cat(lK, dim=0)

    def trainingEnd(self):

        print("")
        print("Sampling results: ")

        model = self.trainer.model
        model.eval()
        with torch.no_grad():
            images = self.images.to(self.trainer.device)
            images = self.trainer.dsTrain.transform(images)
            khat = model(images)

        # Undistore & save
        for i in range(len(self.images)):
            imgOriginal = toNumpyImage(self.images[i])
            k = self.k[i].cpu().detach().numpy()
            kh = khat[i].cpu().detach().numpy()

            imgOriginal = cv2.cvtColor(imgOriginal, cv2.COLOR_RGB2BGR)

            # Undistortneme obrazky
            undistortedLabel = undistort(imgOriginal, k, scaleShape=self.scaleShape)
            undistortedEstimate = undistort(imgOriginal, kh, scaleShape=self.scaleShape)

            # Zmensime ...
            imgOriginal = rescale(imgOriginal, scaleShape=self.scaleShape)
            undistortedLabel = rescale(undistortedLabel, scaleShape=self.scaleShape)
            undistortedEstimate = rescale(undistortedEstimate, scaleShape=self.scaleShape)

            cv2.imwrite(os.path.join(self.folder, "image-{}-original.png".format(i)), imgOriginal)
            cv2.imwrite(os.path.join(self.folder, "image-{}-label.png".format(i)), undistortedLabel)
            cv2.imwrite(os.path.join(self.folder, "image-{}-estimate.png".format(i)), undistortedEstimate)


#------------------------------------------------------------------------------
#   PSNR_SSIM_Sampler
#------------------------------------------------------------------------------

class PSNR_SSIM_Sampler(Logger):
    def __init__(self, trainer):
        self.trainer = trainer
        self.fn = os.path.join(self.trainer.outputFolder, "psnr_ssim.csv")

    def trainingEnd(self):
        print("")
        print("Sampling PSNR, SSIM: ")

        model = torch.nn.DataParallel(self.trainer.model)
        model.eval()

        result = {
            "psnr": 0.0,
            "ssim": 0.0,
            "mdld": 0.0
        }
        nItems = 0

        with torch.no_grad():
            progress = getTqdm(self.trainer.dsVal.loader)
            progress.set_description("PSNR/SSIM")
            for (x, k) in progress:
                x = x.to(self.trainer.device)
                k = k.to(self.trainer.device)
                x = self.trainer.dsTrain.transform(x)

                # Compute the prediction
                k_hat = model(x)
                k_hat = k2FromK1(k_hat)     # compute k2 coefficient

                # Evaluate minibatch
                nItems += self.evalMinibatch(x, k, k_hat, result)

        # Zapiseme do CSVcka
        csv = open(self.fn, "w")
        csv.write("psnr,ssim,mdld" + "\n")
        csv.write("{},{},{}".format(
            result["psnr"] / (nItems),
            result["ssim"] / (nItems),
            result["mdld"] / (nItems)
        ) + "\n")
        csv.flush()
        csv.close()
        print("  PSNR: {}".format(result["psnr"] / (nItems)))
        print("  SSIM: {}".format(result["ssim"] / (nItems)))
        print("  MDLD: {}".format(result["mdld"] / (nItems)))


    def evalMinibatch(self, inX, inK, inKhat, result):
        '''
            1. Pre vsetky obrazky zbehneme
                - Undistortneme podla labelu
                - Undistortneme podla khat
                - PSNR, SSIM
        '''

        b,c,h,w = inX.shape
        mMdld = MDLD(shape=(h,w), device=self.trainer.device)

        # MDLD
        m = mMdld(inKhat, inK).sum()
        result["mdld"] = result["mdld"] + m.cpu().detach().numpy().item()

        for i in range(b):
            x = inX[i]
            k = inK[i].cpu().detach().numpy()
            kh = inKhat[i].cpu().detach().numpy()

            # Skonvertujeme na obrazok
            x = toNumpyImage(x)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

            # Undistortneme obrazky
            undistortedLabel = undistort(x, k)
            undistortedEstimate = undistort(x, kh)

            # Spocitame metriky
            psnr = peak_signal_noise_ratio(undistortedLabel, undistortedEstimate)
            ssim = structural_similarity(undistortedLabel, undistortedEstimate, channel_axis=-1)

            result["psnr"] = result["psnr"] + psnr
            result["ssim"] = result["ssim"] + ssim        

        # Sprocesovali sme B obrazkov
        return b




