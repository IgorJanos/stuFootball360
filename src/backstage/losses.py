#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np




#------------------------------------------------------------------------------
#   MDLD class
#------------------------------------------------------------------------------

def distortionLevel_Polynomial2(RR, k):
    k1 = k[:,0].view(-1, 1, 1)
    k2 = k[:,1].view(-1, 1, 1)
    d = torch.ones_like(RR, device=RR.device)
    d += k1 * (RR ** 2)
    d += k2 * (RR ** 4)
    return d

def distortTorch(RR, k):
    return RR * distortionLevel_Polynomial2(RR, k)



class MDLD(nn.Module):
    def __init__(
        self, 
        shape,
        device, 
        aspect=16.0/9.0,
        distortFunc=distortionLevel_Polynomial2
    ):
        super().__init__()

        # Store helpers
        self.distortFunc = distortFunc
        self.device = device

        # It's enough to compute on one quarter
        h, w = shape
        h = h // 2
        w = w // 2
        x = np.linspace(0, 0.5*aspect, w)
        y = np.linspace(0, 0.5, h)
        xx,yy = np.meshgrid(x,y)

        # Spocitame si maticu druhych mocnin radiusu od stredu
        self.RR = torch.tensor(np.sqrt(xx**2 + yy**2)).unsqueeze(0)     # 1, H, W
        self.RR = self.RR.to(self.device)
        self.mult = 4.0 / (h*w)     


    def __call__(self, yhat, y):     
        '''
            YHAT, Y in shape - (Batch, N)
        '''   
        yhat = yhat.to(self.device)
        y = y.to(self.device)

        B, N = yhat.shape

        # Expand to batch size
        RR = self.RR.repeat(B, 1, 1)

        # Distort
        dyhat = self.distortFunc(RR, yhat)      # (BATCH,H,W)
        dy = self.distortFunc(RR, y)            # (BATCH,H,W)

        # Compute the metric
        a = torch.abs(dyhat - dy)
        a = torch.sum(a, dim=1)                 # Sum over H
        a = torch.sum(a, dim=1, keepdim=True)   # Sum over W (BATCH,1)
        a = a * self.mult
        return a




