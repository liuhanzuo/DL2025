import torch.nn as nn
import torch.nn.functional as F
import torch

class Expert(nn.Module):
    def createManyResBlock(self,channels=64,BlockNum=3,kernel_size=3):
        self.cnt+=1
        manyResBlock = []
        for i in range(BlockNum):
            x = nn.Sequential(
                nn.Conv2d(channels,channels,kernel_size,padding=(kernel_size-1)//2),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                # nn.Dropout2d(0.1),
                nn.Conv2d(channels,channels,kernel_size,padding=(kernel_size-1)//2),
            )
            self.add_module(f'{self.cnt}_{i}',x)
            manyResBlock.append(x)
        return manyResBlock
    def PassThrough(self,manyResBlock:list,x):
        for i in range(len(manyResBlock)):
            x = F.relu(x+manyResBlock[i](x))
            # if i%2:
            #     x = nn.MaxPool2d(2)(x)
        return x

    def __init__(self):
        super(Expert, self).__init__()
        self.cnt=0
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,7,padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.manyResBlock1 = self.createManyResBlock(
            BlockNum=4
        )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.manyResBlock2 = self.createManyResBlock(channels=128
                                                     )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
            )
        self.manyResBlock3 = self.createManyResBlock(channels=128
                                                     )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,64,5,padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
            )
        self.manyResBlock4 = self.createManyResBlock(channels=64
                                                     )
    def forward(self, x):
        bsize = x.shape[0]
        x = self.pre_process(x)
        x = self.conv1(x)
        # print('after conv1',x.shape)
        x = self.PassThrough(self.manyResBlock1,x)
        # print('after block 1',x.shape)
        x = self.conv2(x)
        # print('after conv2',x.shape)
        x = self.PassThrough(self.manyResBlock2,x)
        # print('after block 2',x.shape)
        x = self.conv3(x)
        # print('after conv3',x.shape)
        x = self.PassThrough(self.manyResBlock3,x)
        # print('after block 3',x.shape)
        x = self.conv4(x)
        # print('after conv4',x.shape)
        x = self.PassThrough(self.manyResBlock4,x)
        # print('after block 4',x.shape)
        x = nn.AvgPool2d(4)(x)
        y = x.reshape(bsize,-1)
        return y
    # def to(self,device):
    #     self.final.to(device)
    #     lst1 = [
    #         self.conv1,
    #             self.conv2,
    #             self.conv3,
    #             self.conv4,
    #             ]
    #     for i in lst1:
    #         i.to(device)
    #     lst = [
    #         self.manyResBlock1,
    #         self.manyResBlock2,
    #         self.manyResBlock3,
    #         self.manyResBlock4
    #         ]
    #     for i in lst:
    #         for j in i:
    #             j.to(device)
    def pre_process(self, x):
        return x.float()

class Gate(nn.Module):
    def __init__(self, input_size, num_experts=10, hidden_size=50):
        super(Gate, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_experts)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net(nn.Module):
    '''
    A MoE network 
    '''
    def __init__(self, num_experts = 10, topk = 3, hidden_size = 50):
        super(Net, self).__init__()
        self.experts = [Expert() for i in range(num_experts)]
        self.common_expert = Expert()
        self.num_experts = num_experts
        self.gate = Gate(64, num_experts, hidden_size)
        self.topk = topk
        self.final = nn.Linear(128, 10)
    def forward(self, x):
        # 公共专家处理
        common_out = self.common_expert(x)
        
        # 获取各个专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, 64]
        
        # 门控机制
        gate_weights = self.gate(common_out)  # [B, num_experts]
        
        # 选择topk专家
        topk_weights, topk_indices = torch.topk(
            gate_weights, self.topk, dim=1)  # [B, topk]
        
        # 加权求和
        batch_indices = torch.arange(x.size(0)).unsqueeze(-1)  # [B, 1]
        selected_outputs = expert_outputs[batch_indices, topk_indices]  # [B, topk, 64]
        weighted_outputs = (selected_outputs * topk_weights.unsqueeze(-1)).sum(1)  # [B, 64]
        
        # 合并公共专家和MoE结果
        combined = torch.cat([common_out, weighted_outputs], dim=1)  # [B, 128]
        
        # 最终分类
        return self.final(combined)


    def pre_process(self, x):
        return x.float()