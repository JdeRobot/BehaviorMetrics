import torch
import torch.nn as nn
from .convlstm import ConvLSTM

class DeepestLSTMTinyPilotNet(nn.Module):
    def __init__(self, image_shape, num_labels):
        super(DeepestLSTMTinyPilotNet, self).__init__()
        self.num_channels = image_shape[2]
        self.cn_1 = nn.Conv2d(self.num_channels, 8, kernel_size=3, stride=2)
        self.relu_1 = nn.ReLU()
        self.cn_2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.relu_2 = nn.ReLU()
        self.cn_3 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.relu_3 = nn.ReLU() 
        self.dropout_1 = nn.Dropout(0.2)

        self.clstm_n = ConvLSTM(8, 8, (5, 5), 3, batch_first=True, bias=True, return_all_layers=False)
        
        self.fc_1 = nn.Linear(8*11*5, 50)
        self.relu_fc_1 = nn.ReLU()
        self.fc_2 = nn.Linear(50, 10)
        self.relu_fc_2 = nn.ReLU()
        self.fc_3 = nn.Linear(10, num_labels)

    def forward(self, img):        
        out = self.cn_1(img)
        out = self.relu_1(out)
        out = self.cn_2(out)
        out = self.relu_2(out)
        out = self.cn_3(out)
        out = self.relu_3(out)
        out = self.dropout_1(out)
        # add additional dimension at 1
        out = out.unsqueeze(1) 

        _, last_states = self.clstm_n(out)
        out =  last_states[0][0]  # 0 for layer index, 0 for h index
        # flatten
        out = out.reshape(out.size(0), -1)
        out = self.fc_1(out)
        out = self.relu_fc_1(out)
        out = self.fc_2(out)
        out = self.relu_fc_2(out)
        out = self.fc_3(out)
        return out
