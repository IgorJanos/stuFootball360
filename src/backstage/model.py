#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import torch
import torch.nn as nn

import torchvision



def createDensenet(backbone):    
    if (backbone == "densenet121"):   model = torchvision.models.densenet121(pretrained=True)
    elif (backbone == "densenet161"): model = torchvision.models.densenet161(pretrained=True)
    elif (backbone == "densenet169"): model = torchvision.models.densenet169(pretrained=True)
    elif (backbone == "densenet201"): model = torchvision.models.densenet201(pretrained=True)
    else:
        return None, 0

    numFeatures = model.classifier.in_features
    model.classifier = nn.Sequential()
    return model, numFeatures




#------------------------------------------------------------------------------
#
#   DistortionModel class
#
#------------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "densenet121": createDensenet,
    "densenet161": createDensenet,
    "densenet169": createDensenet,
    "densenet201": createDensenet,
}


class DistortionModel(nn.Module):
    def __init__(self, conf, nOut=1):
        super().__init__()

        backbone = conf["backbone"]
        numHidden = conf["nHidden"]

        # Create backbone feature extractor
        self.extractor, numFeatures = SUPPORTED_MODELS[backbone](backbone)

        # Append regressor head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(numFeatures, numHidden),
            nn.BatchNorm1d(numHidden),
            nn.ReLU(),
            nn.Linear(numHidden, nOut)
        )


    def forward(self, x):
        features = self.extractor(x)
        khat = self.regressor(features)
        return khat



def createModel(conf):
    return None