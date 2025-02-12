import torch
import torch.nn as nn


class PilotNet(nn.Module):
    def __init__(self, image_shape, num_labels):
        super(PilotNet, self).__init__()
       
        self.num_channels = image_shape[2]
        # Batch normalization?
        self.batchnorm_input = nn.BatchNorm2d(self.num_channels)  # Para imágenes en formato RGB (3 canales)
        self.cn_1 = nn.Conv2d(in_channels=self.num_channels, out_channels=24, kernel_size=5, stride=2)
        self.relu_1 = nn.ReLU()
        self.cn_2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.relu_2 = nn.ReLU()
        self.cn_3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.relu_3 = nn.ReLU() 
        self.cn_4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.relu_4 = nn.ReLU() 
        self.cn_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu_5 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()

        # Flatten layer?
        self.fc_1 = nn.Linear(1152, 1164)  # add embedding layer output size 
        self.relu_fc_1 = nn.ReLU()
        self.fc_2 = nn.Linear(1164, 100)
        self.relu_fc_2 = nn.ReLU()
        self.fc_3 = nn.Linear(100, 50)
        self.relu_fc_3 = nn.ReLU()
        self.fc_4 = nn.Linear(50, 10)
        self.relu_fc_4 = nn.ReLU()
        self.fc_5 = nn.Linear(10, num_labels)

    def forward(self, img):
        out = self.batchnorm_input(img)
        out = self.cn_1(img)
        out = self.relu_1(out)
        out = self.cn_2(out)
        out = self.relu_2(out)
        out = self.cn_3(out)
        out = self.relu_3(out)
        
        out = self.cn_4(out)
        out = self.relu_4(out)
        out = self.cn_5(out)
        out = self.relu_5(out)
        
        out = self.dropout_1(out)
            
        #out = out.view(-1, 1152)
        out = self.flatten(out)
        
        out = self.fc_1(out)
        out = self.relu_fc_1(out)
        out = self.fc_2(out)
        out = self.relu_fc_2(out)
        out = self.fc_3(out)
        out = self.relu_fc_3(out)
        out = self.fc_4(out)
        out = self.relu_fc_4(out)
        out = self.fc_5(out)

        #out = torch.sigmoid(out)

        return out