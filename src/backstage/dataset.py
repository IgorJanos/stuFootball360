#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
import cv2
import h5py
import json
import numpy as np
import torch

from torch.utils.data import Dataset
import urllib.request

BASE_DOWNLOAD_URL="https://data.deepmagic.sk/fiit/football360/"
FOOTBALL360_SET_NAMES = {
    "A": "football360-setA.h5",
    "B": "football360-setB.h5",
    "C": "football360-setC.h5",
    "V": "football360-setV.h5",
}

#------------------------------------------------------------------------------
#
#   FootballDataset class
#
#------------------------------------------------------------------------------

class FootballDataset(Dataset):
    def __init__(self, folder, setName, scaleShape=None, asTensor=True):
        self.folder = folder
        self.h5file = None
        self.filename = os.path.join(folder, FOOTBALL360_SET_NAMES[setName])
        self.url = BASE_DOWNLOAD_URL + FOOTBALL360_SET_NAMES[setName]
        self.info = {}
        self.labels = None
        self.scaleShape = scaleShape
        self.len = 0
        self.asTensor = asTensor
        self.assertOpen()

    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        self.assertOpen()
        image, label = self.loadRaw(idx)

        # Convert to torch tensor
        if (self.asTensor):
            image = FootballDataset.toTensor(image)
            
        return image, label


    def assertOpen(self):
        # Try open file
        if (self.h5file is None):       
            # Download first
            if (not os.path.isfile(self.filename)):
                os.makedirs(self.folder, exist_ok=True)
                try:
                    print("Downloading file: ", self.url)
                    urllib.request.urlretrieve(self.url, self.filename)
                except:
                    print("Error downloading: ", self.url)
                    return None

            self.h5file = h5py.File(self.filename, mode="r")
            groupImages = self.h5file["images"]
            self.len = len(groupImages)
            self.info = json.loads(bytes(self.h5file.get("info")))
            self.labels = self.h5file.get("labels")

            if (self.len > 1000): self.len = 1000


    def loadRaw(self, idx):
        groupImages = self.h5file["images"]
        # Loadneme image
        bImage = groupImages.get(str(idx))
        image = cv2.imdecode(np.array(bImage), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (self.scaleShape is not None):
            image = cv2.resize(image, self.scaleShape, interpolation=cv2.INTER_AREA)

        dist = np.array(self.h5file["labels"][idx])
        return image, dist

    @staticmethod
    def toNumpyImage(tx: torch.Tensor):
        y = tx.cpu().detach().numpy()
        y = (255 * y).astype(np.uint8)
        y = np.transpose(y, (1, 2, 0))
        return y

    @staticmethod
    def toTensor(image: np.ndarray):
        image = image.transpose((2, 0, 1))
        image = image.astype('float32') / 255.0
        return torch.from_numpy(image)