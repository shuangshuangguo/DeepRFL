from torch import nn

import bninception
from ops.basic_ops import ConsensusModule
from transforms import *


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Plainblock(nn.Module):
    expansion = 4

    def __init__(self, planes, inplanes1, inplanes2, stride=1):
        super(Plainblock, self).__init__()
        self.conv1 = conv1x1(planes, inplanes1)
        self.bn1 = nn.BatchNorm2d(inplanes1)
        self.conv2 = conv3x3(planes, inplanes2)
        self.bn2 = nn.BatchNorm2d(inplanes2)
        self.conv3 = conv1x1(inplanes2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        out3 = self.conv2(x)
        out3 = self.bn2(out3)
        out = self.relu(out3)
        return out


class RFL(nn.Module):
    def __init__(self, block, num_class, num_segments, modality, init_path='', new_length=None,
                 consensus_type='avg', dropout=0.8, crop_num=1):
        super(RFL, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.dropout = dropout
        self.crop_num = crop_num
        self.init_path = init_path
        if new_length is None:
            self.new_length = 1 if modality == 'RGB' else 5
        else:
            self.new_length = new_length
        self.feature = getattr(bninception, 'bninception')(self.modality, self.init_path)
        self.layer1 = block(1024, 512, 256)
        self.avg = nn.AvgPool2d(7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.drop = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(1024, num_class)
        self.consensus = ConsensusModule(consensus_type)
        self.input_size = 224
        self.input_mean = [104, 117, 128]
        self.input_std = [1]
        if self.modality == 'Flow':
            self.input_mean = [128]
        elif self.modality == "RGBDiff":
            self.input_mean = self.input_mean * (1 + self.new_length)

    def forward(self, x, part=True):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        x = x.view((-1, sample_len) + x.size()[-2:])
        feature = self.feature(x)
        if not part:
            feature = feature.view((-1, self.num_segments) + feature.size()[1:])
            feature = self.consensus(feature).squeeze(1)
            feature = self.avg(feature)
            feature_drop = self.drop(feature)
            feature_drop = feature_drop.view(-1, 1024)
            output = self.fc(feature_drop)
            return output, feature
        feature = feature.view((-1, self.num_segments) + feature.size()[1:])
        feature = self.consensus(feature).squeeze(1)
        feature = self.avg(feature)
        feature = feature.detach()
        residual = feature
        feature_gen = self.layer1(feature)
        feature_gen = feature_gen + residual
        feature_gen_drop = self.drop(feature_gen)
        feature_gen_drop = feature_gen_drop.view(-1, 1024)
        output = self.fc(feature_gen_drop)
        return output, feature_gen

    def train(self, mode=True):
        super(RFL, self).train(mode)
        count = 0
        print("Freezing BatchNorm2D except the first one.")
        for m in self.children():
            for n in m.children():
                if isinstance(n, torch.nn.BatchNorm2d):
                    count += 1
                    if count >= 2:
                        n.eval()
                        n.weight.requires_grad = False
                        n.bias.requires_grad = False
            break

    def get_optim_policies(self):
        weight = []
        bias = []
        bn = []
        fea_weight, fea_bias, fea_bn = [], [], []
        idx = 0
        for m in self.children():
            idx += 1
            if idx == 1:
                bn_cnt = 0
                for n in m.children():
                    if isinstance(n, torch.nn.Conv2d):
                        ps = list(n.parameters())
                        fea_weight.append(ps[0])
                        if len(ps) == 2:
                            fea_bias.append(ps[1])
                    elif isinstance(n, torch.nn.BatchNorm2d):
                        bn_cnt += 1
                        if bn_cnt == 1:
                            fea_bn.extend(list(n.parameters()))
            if 3 > idx > 1:
                print(m)
                for n in m.modules():
                    if isinstance(n, torch.nn.Conv2d):
                        number = n.kernel_size[0] * n.kernel_size[1] * n.out_channels
                        n.weight.data.normal_(0, math.sqrt(2. / number))
                        ps = list(n.parameters())
                        weight.append(ps[0])
                        if len(ps) == 2:
                            bias.append(ps[1])
                    elif isinstance(n, torch.nn.BatchNorm2d):
                        n.weight.data.fill_(1)
                        n.bias.data.zero_()
                        bn.extend(list(n.parameters()))
                    else:
                        pass
            if idx == 5:
                print(m)
                assert isinstance(m, torch.nn.Linear)
                torch.nn.init.normal_(m.weight, 0, 0.001)
                torch.nn.init.constant_(m.bias, 0)
                ps = list(m.parameters())
                weight.append(ps[0])
                if len(ps) == 2:
                    bias.append(ps[1])
        return [
            {'params': fea_weight, 'lr_mult': 0.01, 'decay_mult': 1,
             'name': "fea_weight"},
            {'params': fea_bias, 'lr_mult': 0.02, 'decay_mult': 0,
             'name': "fea_bias"},
            {'params': fea_bn, 'lr_mult': 0.01, 'decay_mult': 0,
             'name': "fea BN scale/shift"},
            {'params': weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"}]

    def get_augmentation(self):
        modality = self.modality
        if modality == 'RGB':
            return torchvision.transforms.Compose(
                [GroupMultiScaleCrop(224, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])
        elif modality == 'Flow':
            return torchvision.transforms.Compose(
                [GroupMultiScaleCrop(224, [1, .875, .75]), GroupRandomHorizontalFlip(is_flow=True)])

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224


def RFL_model(num_classes, num_segments, modality, init_path='', dropout=0.5):
    model = RFL(Plainblock, num_classes, num_segments, modality, init_path=init_path, new_length=None,
                consensus_type='avg', dropout=dropout, crop_num=1)
    return model


if __name__ == '__main__':
    init_path = '/home/gss/Code/Action_Recognition/caffe2pytorch-tsn/model/ucf101_rgb.pth'
    model = RFL_model(101, 3, 'RGB', init_path)
