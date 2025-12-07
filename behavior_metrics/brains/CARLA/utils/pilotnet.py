import torch
import torch.nn as nn


class PilotNet(nn.Module):
    def __init__(self,
                image_shape,
                num_labels):
        super().__init__()

        self.img_height = image_shape[0]
        self.img_width = image_shape[1]
        self.num_channels = 3

        self.output_size = num_labels
        
        self.ln_1 = nn.BatchNorm2d(self.num_channels, eps=1e-03)

        self.cn_1 = nn.Conv2d(self.num_channels, 24, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU()
        self.cn_2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.relu2 = nn.ReLU()
        self.cn_3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.relu3 = nn.ReLU()
        self.cn_4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.relu4 = nn.ReLU()
        self.cn_5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu5 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(1 * 18 * 64, 1164)
        self.relu_fc1 = nn.ReLU()
        self.fc_2 = nn.Linear(1164, 100)
        self.relu_fc2 = nn.ReLU()
        self.fc_3 = nn.Linear(100, 50)
        self.relu_fc3 = nn.ReLU()
        self.fc_4 = nn.Linear(50, 10)
        self.relu_fc4 = nn.ReLU()
        self.fc_5 = nn.Linear(10, self.output_size)

    def forward(self, img):

        out = self.ln_1(img)

        out = self.cn_1(out)
        out = self.relu1(out)
        out = self.cn_2(out)
        out = self.relu2(out)
        out = self.cn_3(out)
        out = self.relu3(out)
        out = self.cn_4(out)
        out = self.relu4(out)
        out = self.cn_5(out)
        out = self.relu5(out)

        out = self.flatten(out)

        out = self.fc_1(out)
        out = self.relu_fc1(out)
        out = self.fc_2(out)
        out = self.relu_fc2(out)
        out = self.fc_3(out)
        out = self.relu_fc3(out)
        out = self.fc_4(out)
        out = self.relu_fc4(out)
        out = self.fc_5(out)

        return out