#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import torch
import cv2
import numpy as np
from tqdm import tqdm



class AverageCounter:
    def __init__(self):
        self.reset()

    def step(self, value):
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0


class Statistics:
    def __init__(self, listNames):
        self.items = {}
        for k in listNames:
            self.items[k] = AverageCounter()

    def step(self, k, value):
        self.items[k].step(value)

    def reset(self):
        for k in self.items:
            self.items[k].reset()

    def getAvg(self, listNames=None):
        result = {}
        if (listNames is None):
            listNames = self.items.keys()
        for k in listNames:
            result[k] = self.items[k].avg
        return result



def k2FromK1(k1):
    k2 = 0.019*k1 + 0.805*(k1**2)
    return torch.cat([k1, k2], dim=1)


def getTqdm(data, ncols=None):
    return tqdm(
                data, leave=False, 
                bar_format='{l_bar}{bar:20}{r_bar}',
                ncols=ncols,
                ascii=True
                )

def rescale(image, scaleShape=(640,360)):
    return cv2.resize(image, scaleShape)


def buildMaps(size, k):
    h, w = size
    cx = w/2.0
    cy = h/2.0

    # Ideme zlozit mapy
    x = (np.linspace(0, w+1, w) - cx) / h
    y = (np.linspace(0, h+1, h) - cy) / h
    xx,yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)

    # Nascitavame mocniny koeficientov
    d = np.ones_like(rr)
    for i,kk in enumerate(k):
        d += kk * (rr ** (2 * (i+1)))

    mapX = (d*xx*h + cx).astype(np.float32)
    mapY = (d*yy*h + cy).astype(np.float32)
    return mapX, mapY

def undistort(image: np.ndarray, k:np.ndarray, scaleShape=(640, 360)):
    image = rescale(image, scaleShape)

    # Pouzivame skalovaci faktor -> H ~ 1.0
    w, h = image.shape[1], image.shape[0]

    # Create undistort maps
    k1 = k[0]
    if (k.shape[0] > 1):
        k2 = k[1]
    else:
        k2 = 0.019*k1 + 0.805*(k1 ** 2)
    kk = [k1, k2]

    map1, map2 = buildMaps((h,w), kk)

    # Undistort image
    result = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    return result
