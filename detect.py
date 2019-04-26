import sys

import torch

from shufflenet import ShuffleNet

net = ShuffleNet(128, 3, 196)
net.cuda()
net.load_state_dict(torch.load(sys.argv[1]))
net.cpu()
torch.save(net.state_dict(), "cpu.pth")
