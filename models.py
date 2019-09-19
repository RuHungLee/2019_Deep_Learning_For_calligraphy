import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
# from torchvision.models.resnet import resnet50
class Model(nn.Module):

    def __init__(self, dropout=0):
        super().__init__()
        resnet = resnet50()
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,

            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(2048*3*3, 1022)
        self.rnn = nn.GRU(1024, 1024, dropout=dropout)
        self.fc2 = nn.Linear(1024, 2)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, targets=None):
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = self.fc1(x)
        features = x.view((1, 1, -1))
        y = torch.zeros((1, 1, 2), device='cuda:0')

        outputs = []
        num_steps = 80
        if targets:
            for timestep in range(num_steps):
                pass
        else:
            for timestep in range(num_steps):
                x = torch.cat([features, y], dim=2)     # shape: 1, 1, 1024
                h = torch.zeros((1, 1, 1024), device='cuda:0')           # shape: 1, 1, 1024
                x, h = self.rnn(x, h)
                y = self.fc2(x)
                outputs.append(y)

        outputs = torch.stack(outputs)
        return outputs


class Model_v2(nn.Module):

    def __init__(self, dropout=0):
        super().__init__()
        resnet = resnet50()
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,

            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048*3*3, 1024*2),
            nn.Linear(1024*2, 1024*2),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x, targets=None):
        x = self.cnn(x)
        predict = self.classifier(x).view(1024, 2)
        # predict.squeeze_()
        loss = self.loss_fn(predict, targets) \
            if targets is not None \
            else None
        return predict, loss


class Model_v3(nn.Module):

    def __init__(self, dropout=0):
        super().__init__()
        resnet = resnet50()
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048*3*3, 1024*2),
            nn.Linear(1024*2, 1024*2),
            nn.Linear(1024*2, 1024*2),
            nn.Linear(1024*2, 1024*2),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x, targets=None):
        x = self.cnn(x)
        x = x.view(-1 , 2048*3*3)
        predict = self.classifier(x).view(-1 , 1024 , 2) 
        loss = self.loss_fn(predict, targets) 
        return predict, loss
