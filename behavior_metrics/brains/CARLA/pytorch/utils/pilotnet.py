import torch
import torch.nn as nn


class PilotNet(nn.Module):
    def __init__(self,
                image_shape,
                num_labels):
        super(PilotNet, self).__init__()

        self.img_height = image_shape[0]
        self.img_width = image_shape[1]
        self.num_channels = image_shape[2]

        self.output_size = num_labels
        
        self.ln_1 = nn.BatchNorm2d(self.num_channels, eps=1e-03)

        self.cn_1 = nn.Conv2d(self.num_channels, 24, kernel_size=5, stride=2)
        self.cn_2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.cn_3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.cn_4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.cn_5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc_1 = nn.Linear(1 * 18 * 64, 1164)
        self.fc_2 = nn.Linear(1164, 100)
        self.fc_3 = nn.Linear(100, 50)
        self.fc_4 = nn.Linear(50, 10)
        self.fc_5 = nn.Linear(10, self.output_size)

    def forward(self, img):

        out = self.ln_1(img)

        out = self.cn_1(out)
        out = torch.relu(out)
        out = self.cn_2(out)
        out = torch.relu(out)
        out = self.cn_3(out)
        out = torch.relu(out)
        out = self.cn_4(out)
        out = torch.relu(out)
        out = self.cn_5(out)
        out = torch.relu(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc_1(out)
        out = torch.relu(out)
        out = self.fc_2(out)
        out = torch.relu(out)
        out = self.fc_3(out)
        out = torch.relu(out)
        out = self.fc_4(out)
        out = torch.relu(out)
        out = self.fc_5(out)

        return out
