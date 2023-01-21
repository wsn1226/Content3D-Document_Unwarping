import torch.nn as nn
import math
import torch


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        #print(x.shape)
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DecoderConvLayer(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net=nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ConvTranspose2d(in_channels=input_channels,out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False),
        nn.ReLU()
        )
    
    def forward(self,x):
        return self.net(x)

class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=960, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        #print(input_channel)
        layers = [conv_3x3_bn(4, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        #self.conv2 = conv_3x3_bn()
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        #self.classifier = nn.Sequential(
            #nn.Linear(exp_size, output_channel),
            #h_swish(),
            #nn.Dropout(0.2),
            #nn.Linear(output_channel, num_classes),
        #)

        upconv=[]
        upconv.append(DecoderConvLayer(960,160,1,1))
        self.conv2=conv_3x3_bn(320,160)
        upconv.append(DecoderConvLayer(160,112,2,2))
        self.conv3=conv_3x3_bn(224,112)
        upconv.append(DecoderConvLayer(112,80,1,1))
        self.conv4=conv_3x3_bn(160,80)
        upconv.append(DecoderConvLayer(80,40,2,2))
        self.conv5=conv_3x3_bn(80,40)
        upconv.append(DecoderConvLayer(40,24,2,2))
        self.conv6=conv_3x3_bn(48,24)
        upconv.append(DecoderConvLayer(24,16,2,2))
        self.conv7=conv_3x3_bn(32,16)
        upconv.append(nn.ConvTranspose2d(16,2,2,2))

        self.upconvs=nn.Sequential(*upconv)
        self.finalactivation=nn.Hardtanh()

        self._initialize_weights()

    def forward(self, x):
        skip_connections=[]
        for i in range(len(self.features)):
            x=self.features[i](x)
            if i==1 or i==3 or i==6 or i==10 or i==12 or i==15:
                skip_connections.append(x)
                #print(x.shape)

        x = self.conv(x) #torch.Size([1, 960, 4, 4])
        #skip_connections.append(x)
        #x = torch.reshape(x, (960, 16)).permute(1,0)
        #x = self.avgpool(x) #torch.Size([1, 960, 1, 1])
        #skip_connections.append(x)
        #print(x.shape)
        skip=skip_connections[::-1]

        #x = x.view(x.size(0), -1)
        #x = self.classifier(x).permute(1,0)
        #x = torch.reshape(x,(960,4,4)).unsqueeze(0)
        x=self.upconvs[0](x)
        x=torch.cat([x,skip[0]],dim=1)
        x=self.conv2(x)

        x=self.upconvs[1](x)
        x=torch.cat([x,skip[1]],dim=1)
        x=self.conv3(x)

        x=self.upconvs[2](x)
        x=torch.cat([x,skip[2]],dim=1)
        x=self.conv4(x)

        x=self.upconvs[3](x)
        x=torch.cat([x,skip[3]],dim=1)
        x=self.conv5(x)

        x=self.upconvs[4](x)
        x=torch.cat([x,skip[4]],dim=1)
        x=self.conv6(x)

        x=self.upconvs[5](x)
        x=torch.cat([x,skip[5]],dim=1)
        x=self.conv7(x)
        
        x=self.upconvs[6](x)
        x=self.finalactivation(x)

        #print(x.shape)        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':
    img=torch.randn(1,4,128,128)
    model = mobilenetv3_large()
    pred=model(img)
    print(pred.shape)
    print(torch.max(pred),torch.min(pred))
    #print((pred<0).to(torch.uint8).nonzero())
    #print((pred==-1).to(torch.uint8).nonzero())
    print(count_parameters(model))
