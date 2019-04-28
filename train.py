import sys

import torch
import torch.nn as nn
import torch.optim as optim

from shufflenet import ShuffleNet, ConvNet
from preprocess import WFLW, Transformer



def train(epochs):
    # input settings
    image_size = 128
    # network settings
    net = ConvNet(image_size, 3, 196)
    net.cuda()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    if len(sys.argv) == 2:
        net.load_state_dict(torch.load(sys.argv[1]))


    # transformer settings
    ts = Transformer(image_size, image_size)

    for epoch_i in range(epochs):
        dataset = WFLW('data/WFLW') # buggy
        trainset = dataset.data_generator(128, 'train')
        for batch_i, (X_train, y_train) in enumerate(trainset):
            ts_data = list(map(ts.letterbox, X_train, y_train))
            X_train = torch.FloatTensor([ts.transform(x[0]) for x in ts_data]).cuda()
            y_train = torch.FloatTensor([x[1] for x in ts_data]).cuda()

            net.train()
            outputs = net(X_train)
            loss = loss_fn(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Train Epoch {epoch_i} Batch {batch_i}, Train Loss: {loss:.3f}")

        dataset = WFLW('data/WFLW') # buggy
        trainset = dataset.data_generator(64, 'valid')
        for batch_i, (X_train, y_train) in enumerate(trainset):
            ts_data = list(map(ts.letterbox, X_train, y_train))
            X_train = torch.FloatTensor([ts.transform(x[0]) for x in ts_data]).cuda()
            y_train = torch.FloatTensor([x[1] for x in ts_data]).cuda()

            net.train()
            outputs = net(X_train)
            loss = loss_fn(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Valid Epoch {epoch_i} Batch {batch_i}, Train Loss: {loss:.3f}")


        torch.save(net.state_dict(), "last.pth")


if __name__ == '__main__':
    train(400)
