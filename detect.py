import sys

import cv2
import numpy as np
import torch

from shufflenet import ShuffleNet
from preprocess import Transformer
from vutils import plot

image_size = 128

net = ShuffleNet(image_size, 3, 196)
net.load_state_dict(torch.load('cpu.pth'))
net.eval()

# preprocess
ts = Transformer(image_size, image_size)
landmark = np.array([1] * 196, dtype='float')
image = cv2.imread('imgs/smile2.png')
image_, _, (scale, x_offset, y_offset) = ts.letterbox(image, landmark)
image_ = ts.transform(image_)
image_ = torch.FloatTensor(image_)
image_ = image_[None, :, :, :]

with torch.no_grad():
    outputs = net(image_)
outputs = outputs.numpy().squeeze()
true_out = ts.inverse_letterbox(outputs, scale, x_offset, y_offset)
plot(image, true_out, 'test.png')