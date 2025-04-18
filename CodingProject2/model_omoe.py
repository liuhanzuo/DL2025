import torch.nn as nn
import torch.nn.functional as F
import torch
class Net(nn.Module):
    def createManyResBlock(self, channels=64, BlockNum=3, kernel_size=3):
        self.cnt += 1
        manyResBlock = []
        for i in range(BlockNum):
            x = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                # nn.Dropout2d(0.1),
                nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size-1)//2),
            )
            self.add_module(f'{self.cnt}_{i}', x)
            manyResBlock.append(x)
        return manyResBlock

    def PassThrough(self, manyResBlock: list, x):
        for i in range(len(manyResBlock)):
            x = F.relu(x + manyResBlock[i](x))
            # if i % 2:
                # x = nn.MaxPool2d(2)(x)
        return x

    def __init__(self):
        super(Net, self).__init__()
        self.cnt = 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.manyResBlock11 = self.createManyResBlock(
            channels=64, kernel_size=5
        )
        self.manyResBlock12 = self.createManyResBlock(
            channels=64, kernel_size=3
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.manyResBLock20 = self.createManyResBlock(
            channels=128, kernel_size=7
        )
        self.manyResBlock21 = self.createManyResBlock(
            channels=128, kernel_size=5
        )
        self.manyResBlock22 = self.createManyResBlock(
            channels=128, kernel_size=3
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 * 3, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.manyResBlock30 = self.createManyResBlock(
            channels=128, kernel_size=7
        )
        self.manyResBlock31 = self.createManyResBlock(channels=128, kernel_size=5)
        self.manyResBlock32 = self.createManyResBlock(channels=128, kernel_size=3)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.manyResBlock4 = self.createManyResBlock(channels=64)
        self.final = nn.Linear(64, 10)

    def forward(self, x):
        bsize = x.shape[0]
        x = self.pre_process(x)
        # x has shape [bsize, 3, 128, 128]
        # print the device of the tensor
        # print(x.device)
        x = self.conv1(x)
        # x has shape [bsize, 64, 64, 64]
        # print('after conv1', x.shape)
        x1 = self.PassThrough(self.manyResBlock11, x)
        x2 = self.PassThrough(self.manyResBlock12, x)
        x = torch.cat([x1, x2], dim=1)
        # x has shape [bsize, 64, 64, 64]
        # print('after block 1', x.shape)
        x = self.conv2(x)
        # x has shape [bsize, 128, 32, 32]
        # print('after conv2', x.shape)
        x0 = self.PassThrough(self.manyResBLock20, x)
        x1 = self.PassThrough(self.manyResBlock21, x)
        x2 = self.PassThrough(self.manyResBlock22, x)
        x = torch.cat([x0, x1, x2], dim=1)
        # x has shape [bsize, 128, 32, 32]
        # print('after block 2', x.shape)
        x = self.conv3(x)
        # x has shape [bsize, 128, 16, 16]
        # print('after conv3', x.shape)
        x0 = self.PassThrough(self.manyResBlock30, x)
        x1 = self.PassThrough(self.manyResBlock31, x)
        x2 = self.PassThrough(self.manyResBlock32, x)
        x = torch.cat([x0, x1, x2], dim=1)
        # x has shape [bsize, 128, 16, 16]
        # print('after block 3', x.shape)
        x = self.conv4(x)
        # x has shape [bsize, 64, 8, 8]
        # print('after conv4', x.shape)
        x = self.PassThrough(self.manyResBlock4, x)
        # print('after block 4', x.shape)
        x = nn.AvgPool2d(4)(x)
        y = x.reshape(bsize, -1)
        y = self.final(y)
        return y, 0 ,0

    def to(self, device):
        lst1 = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.final
        ]
        for i in lst1:
            i.to(device)
        lst = [
            self.manyResBlock11,
            self.manyResBlock12,
            self.manyResBLock20,
            self.manyResBlock21,
            self.manyResBlock22,
            self.manyResBlock30,
            self.manyResBlock31,
            self.manyResBlock32,
            self.manyResBlock4
        ]
        for i in lst:
            for j in i:
                j.to(device)

    def pre_process(self, x):
        return x.float()
