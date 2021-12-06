import os
import copy
import numpy as np
import cv2

import torch
from torch.autograd import Variable
from torchvision import models


def getSampleParams():
    pretrained = models.alexnet(pretrained=True)
    return pretrained

