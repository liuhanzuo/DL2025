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
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size-1)//2),
            )
            self.add_module(f'{self.cnt}_{i}', x)
            manyResBlock.append(x)
        return manyResBlock

    def PassThrough(self, manyResBlock: list, x):
        sequence = [ x ]
        for i in range(len(manyResBlock)):
            for j in range(i):
                x = F.mish(x + sequence[j])
            x = F.mish(x + manyResBlock[i](x))
            sequence.append(x)
        return x

    def __init__(self):
        super(Net, self).__init__()
        self.cnt = 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.manyResBlock11 = self.createManyResBlock(
            channels=64, kernel_size=5, BlockNum=4
        )
        self.manyResBlock12 = self.createManyResBlock(
            channels=64, kernel_size=3, BlockNum=4
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.manyResBlock2 = self.createManyResBlock(
            channels=128, kernel_size=3, BlockNum=4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.manyResBlock3 = self.createManyResBlock(channels=128, kernel_size=3, BlockNum=4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.manyResBlock4 = self.createManyResBlock(channels=64)
        self.final = nn.Sequential(
            nn.Linear(256,256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )
        self.linear0= nn.Sequential(
            nn.Linear(8192, 32),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(8192, 32),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.linear4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

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
        x = self.PassThrough(self.manyResBlock2, x)
        # x has shape [bsize, 128, 32, 32]
        # print('after block 2', x.shape)
        x = self.conv3(x)
        x0 = x.reshape(bsize, -1)
        x0 = self.linear0(x0)
        # x has shape [bsize, 128, 16, 16], x0 has shape [bsize, 128*16*16]
        # print('after conv3', x.shape)
        x = self.PassThrough(self.manyResBlock3, x)
        x1 = x.reshape(bsize, -1)
        x1 = self.linear1(x1)
        # x has shape [bsize, 128, 16, 16]
        # print('after block 3', x.shape)
        x = self.conv4(x)
        x2 = x.reshape(bsize, -1)
        x2 = self.linear2(x2)
        # x has shape [bsize, 64, 8, 8]
        # print('after conv4', x.shape)
        x = self.PassThrough(self.manyResBlock4, x)
        x3 = x.reshape(bsize, -1)
        x3 = self.linear3(x3)
        # print('after block 4', x.shape)
        x = nn.AvgPool2d(4)(x)
        x4 = x.reshape(bsize, -1)
        x4 = self.linear4(x4)
        x = torch.cat([x0, x1, x2, x3, x4],dim=1)
        # print('after concat', x.shape)

        y = x.reshape(bsize, -1)
        y = self.final(y)
        return y

    def to(self, device):
        lst1 = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.linear0,
            self.linear1,
            self.linear2,
            self.linear3,
            self.linear4,
            self.final
        ]
        for i in lst1:
            i.to(device)
        lst = [
            self.manyResBlock11,
            self.manyResBlock12,
            self.manyResBlock2,
            self.manyResBlock3,
            self.manyResBlock4
        ]
        for i in lst:
            for j in i:
                j.to(device)

    def pre_process(self, x):
        return x.float()
