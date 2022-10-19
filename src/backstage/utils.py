#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import cv2
import numpy as np


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
