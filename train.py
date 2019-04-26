import sys

import torch
import torch.nn as nn
import torch.optim as optim

from shufflenet import ShuffleNet
from preprocess import WFLW, Transformer



def train(epochs):
    # input settings
    image_size = 224
    # network settings
    net = ShuffleNet(image_size, 3, 196)
    net.cuda()
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    minloss = 99999
    if len(sys.argv) == 2:
        net.load_state_dict(torch.load(sys.argv[1]))
        minloss = float(sys.argv.split('_')[-1][:-4])


    # dataset settings
    dataset = WFLW('data/WFLW')
    validset = dataset.data_generator(100, 'valid')
    # transformer settings
    ts = Transformer(image_size, image_size)

    for epoch_i in range(epochs):
        trainset = dataset.data_generator(32, 'train')
        for batch_i, (X_train, y_train) in enumerate(trainset):
            print(X_train[0].shape)
            ts_data = list(map(ts.letterbox, X_train, y_train))
            X_train = torch.FloatTensor(ts.transform(x[0]) for x in ts_data).cuda()
            y_train = torch.FloatTensor(x[1] for x in ts_data).cuda()

            net.train()
            outputs = net(X_train)
            loss = loss_fn(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch_i} Batch {batch_i}, Train Loss: {minloss:.3f}")

            if batch_i > 0 and batch_i % 40 == 0:
                X_valid, y_valid = next(validset)
                ts_data = list(map(ts.letterbox, X_valid, y_valid))
                X_valid = torch.FloatTensor(ts.transform(x[0]) for x in ts_data).cuda()
                y_valid = torch.FloatTensor(x[1] for x in ts_data).cuda()

                with torch.no_grad():
                    net.eval()
                    outputs = net(X_valid)
                    loss = loss_fn(outputs, y_valid)
                    if loss < minloss:
                        minloss = loss
                        torch.save(net.state_dict(), f"in_{image_size}_loss_{minloss:.3f}.pth")
                    print(f"Valid Loss: {minloss:.3f}")

if __name__ == '__main__':
    train(2)
