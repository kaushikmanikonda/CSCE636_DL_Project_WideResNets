import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class base_wrn_block(nn.Module):
    def __init__(self, kernels_in, kernels_out, stride, dropRate=0.0):
        super(base_wrn_block, self).__init__()

        # create the basic block of the wideresnet

        self.bn1 = nn.BatchNorm2d(kernels_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(kernels_in, kernels_out, kernel_size=3, stride=stride, padding=1, bias=False)

        # self.droprate = dropRate
        self.dropout = nn.Dropout(p=dropRate)

        self.bn2 = nn.BatchNorm2d(kernels_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(kernels_out, kernels_out, kernel_size=3, stride=1, padding=1, bias=False)

        # for identity connections

        self.in_equals_out = (kernels_in == kernels_out)
        if self.in_equals_out:
            self.convShortcut = None
        else:
            self.convShortcut = nn.Conv2d(kernels_in, kernels_out, kernel_size=1, stride=stride, padding=0, bias=False)


    def forward(self, x):
        if self.in_equals_out:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))

        out = self.dropout(out)
        out = self.conv2(out)

        if self.in_equals_out:
            return torch.add(x, out)
        else:
            return torch.add(self.convShortcut(x), out)


class wrn_block(nn.Module):
    def __init__(self, num_layers, kernels_in, kernels_out, block, stride, dropRate=0.0):
        super(wrn_block, self).__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(block(kernels_in, kernels_out, stride, dropRate))
            else:
                layers.append(block(kernels_out, kernels_out, 1, dropRate))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class MyNetwork(nn.Module):

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.bias.data.zero_()


    def __init__(self, config):
        super(MyNetwork, self).__init__()

        depth = config["depth"]
        num_classes = config["num_classes"]
        width_multiplier = config["width_multiplier"]
        dropRate = config["dropRate"]

        num_features = [16, 16*width_multiplier, 32*width_multiplier, 64*width_multiplier]

        n = (depth - 4) // 6
        block = base_wrn_block

        self.conv1 = nn.Conv2d(3, num_features[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = wrn_block(n, num_features[0], num_features[1], block, 1, dropRate)
        self.block2 = wrn_block(n, num_features[1], num_features[2], block, 2, dropRate)
        self.block3 = wrn_block(n, num_features[2], num_features[3], block, 2, dropRate)

        self.bn1 = nn.BatchNorm2d(num_features[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_features[3], num_classes)
        self.num_features = num_features[3]
        self.init_weights()


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_features)
        out = self.fc(out)
        return out
