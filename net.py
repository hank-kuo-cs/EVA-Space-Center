import torch.nn as nn


architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.network = self._make_network()
        self.regression1 = nn.Linear(512, 256)
        self.regression2 = nn.Linear(256, 128)
        self.regression3 = nn.Linear(128, 64)
        self.regression4 = nn.Linear(64, 3)

    def forward(self, x):
        out = self.network(x)

        out = out.view(out.size(0), -1)

        features = out.clone()

        out = self.regression1(out)
        out = self.regression2(out)
        out = self.regression3(out)
        out = self.regression4(out)

        return features, out

    def _make_network(self):
        layers = []
        in_channels = 1

        for layer in architecture:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, layer, kernel_size=3, padding=1),
                           nn.BatchNorm2d(layer),
                           nn.ReLU(inplace=True)]
                in_channels = layer

        layers += [nn.AvgPool2d(kernel_size=(9, 12))]

        return nn.Sequential(*layers)
