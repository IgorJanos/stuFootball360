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

def createResnet(backbone):
    if (backbone == "resnet18"): model = torchvision.models.resnet18(pretrained=True)
    elif (backbone == "resnet34"): model = torchvision.models.resnet34(pretrained=True)
    elif (backbone == "resnet50"): model = torchvision.models.resnet50(pretrained=True)
    elif (backbone == "resnet101"): model = torchvision.models.resnet101(pretrained=True)
    elif (backbone == "resnet152"): model = torchvision.models.resnet152(pretrained=True)
    else:
        return None, 0

    numFeatures = model.classifier.in_features
    model.classifier = nn.Sequential()
    return model, numFeatures

def createEfficientnet(backbone):
    if (backbone == "efficientnet_b0"): model = torchvision.models.efficientnet_b0(pretrained=True)
    elif (backbone == "efficientnet_b1"): model = torchvision.models.efficientnet_b1(pretrained=True)
    elif (backbone == "efficientnet_b2"): model = torchvision.models.efficientnet_b2(pretrained=True)
    elif (backbone == "efficientnet_b3"): model = torchvision.models.efficientnet_b3(pretrained=True)
    elif (backbone == "efficientnet_b4"): model = torchvision.models.efficientnet_b4(pretrained=True)
    elif (backbone == "efficientnet_b5"): model = torchvision.models.efficientnet_b5(pretrained=True)
    elif (backbone == "efficientnet_b6"): model = torchvision.models.efficientnet_b6(pretrained=True)
    elif (backbone == "efficientnet_b7"): model = torchvision.models.efficientnet_b7(pretrained=True)
    else:
        return None, 0

    numFeatures = model.classifier[1].in_features
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

    "resnet18": createResnet,
    "resnet34": createResnet,
    "resnet50": createResnet,
    "resnet101": createResnet,
    "resnet152": createResnet,

    "efficientnet_b0": createEfficientnet,
    "efficientnet_b1": createEfficientnet,
    "efficientnet_b2": createEfficientnet,
    "efficientnet_b3": createEfficientnet,
    "efficientnet_b4": createEfficientnet,
    "efficientnet_b5": createEfficientnet,
    "efficientnet_b6": createEfficientnet,
    "efficientnet_b7": createEfficientnet
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