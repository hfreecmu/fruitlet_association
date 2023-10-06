import torch.nn as nn

#TODO leaky or non leaky?
def conv(in_channels, out_channels, relu, norm, dropout, kernel_size, stride, padding=1):
    layers = []

    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    layers.append(conv_layer)

    if norm == "instance":
        layers.append(nn.InstanceNorm2d(out_channels))
    elif norm is None:
        pass
    else:
        raise RuntimeError('Illagel norm passed: ' + norm)
    
    if relu:
        layers.append(nn.ReLU())

    if dropout:
        layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)

#TODO add or cat?
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, conv_out_channels, relu, norm, dropout, mode="add"):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=in_channels, out_channels=conv_out_channels, relu=relu, norm=norm, dropout=dropout, kernel_size=3, stride=1)
        self.mode = mode

    def forward(self, x):
        if self.mode == "cat":
            out = torch.cat([x, self.conv_layer(x)], 1)
        elif self.mode == "add":
            out = x + self.conv_layer(x)
        else:
            raise RuntimeError("Illegal mode: " + self.mode)
        return out

class DescriptorEncoder(nn.Module):
    def __init__(self, num_res=3, num_res_dropout=1, norm='instance'):
        super(DescriptorEncoder, self).__init__()

        self.conv_in = conv(256, 512, relu=True, norm=norm, dropout=True, kernel_size=4, stride=2)

        resnets = []
        for i in range(num_res):
            dropout = (i < num_res_dropout)
            resnets.append(ResnetBlock(512, 512, relu=True, norm=norm, dropout=dropout))
        self.resnets = nn.Sequential(*resnets)

        self.conv_out = conv(512, 511, relu=False, norm=None, dropout=False, kernel_size=4, stride=2)

    def forward(self, x):

        x = self.conv_in(x)
        x = self.resnets(x)
        x = self.conv_out(x)

        return x

class KeypointEncoder(nn.Module):
    def __init__(self, norm='instance'):
        super(KeypointEncoder, self).__init__()

        self.conv0 = conv(3, 32, relu=True, norm=norm, dropout=False, kernel_size=4, stride=2)
        self.conv1 = conv(32, 64, relu=True, norm=norm, dropout=True, kernel_size=4, stride=2)
        self.conv2 = conv(64, 128, relu=True, norm=norm, dropout=True, kernel_size=4, stride=2)
        self.conv3 = conv(128, 256, relu=True, norm=norm, dropout = False, kernel_size=4, stride=2)
        self.conv4 = conv(256, 256, relu=True, norm=norm, dropout = False, kernel_size=4, stride=2)
        self.conv5 = conv(256, 511, relu=False, norm=None, dropout = False, kernel_size=4, stride=2, padding=0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


