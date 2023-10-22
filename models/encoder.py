import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm='batch', init_zero_weights=False,
         dropout=False, bias=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))


    if dropout:
        layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)

#TODO add seg as 4th channel???
class DescriptorEncoder(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=32, norm='instance'):
        super(DescriptorEncoder, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################
        self.conv1 = conv(3, conv_dim, 4, stride=2, padding=1, norm=norm, init_zero_weights=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4, stride=2, padding=1, norm=norm, init_zero_weights=False,
                          dropout=False)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, stride=2, padding=1, norm=norm, init_zero_weights=False,
                          dropout=False)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4, stride=2, padding=1, norm=norm, init_zero_weights=False,
                          dropout=False)
        self.conv5 = conv(conv_dim*8, conv_dim*8, 4, stride=2, padding=1, norm=norm, init_zero_weights=False)
        self.conv6 = conv(conv_dim*8, conv_dim*8, 4, stride=2, padding=1, norm=None, init_zero_weights=False, bias=True)

    def forward(self, x):

        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x

#TODO make this harmonic like nerf?
class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, F_dim: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)

