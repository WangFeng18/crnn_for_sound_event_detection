import torch
import torch.nn as nn
import torch.nn.functional as F
import parameter

def conv5x5(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 5, padding=2, bias=False)
class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.rnn_length = int(self.config.segment_length*1000 / parameter.hop_length)
        self.conv1 = conv5x5(1, 96)
        self.bn1 = nn.BatchNorm2d(96)
        self.dp1 = nn.Dropout2d(p=0.25)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        self.conv2 = conv5x5(96, 96)
        self.bn2 = nn.BatchNorm2d(96)
        self.dp2 = nn.Dropout2d(p=0.25)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.conv3 = conv5x5(96, 96)
        self.bn3 = nn.BatchNorm2d(96)
        self.dp3 = nn.Dropout2d(p=0.25)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.rnn = nn.GRU(input_size=96, hidden_size=96, num_layers=1, batch_first=True, dropout=0, bidirectional=config.bidirectional)
        self.fc = nn.Linear(96*(2 if config.bidirectional else 1), parameter.n_classes)

    def forward(self, data):
        out = F.relu(self.bn1(self.conv1(data)))
        out = self.dp1(out)
        out = self.pool1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dp2(out)
        out = self.pool2(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.dp3(out)
        out = self.pool3(out)
        out = out.squeeze(dim=-1).transpose(1,2)
        out, hidden = self.rnn(out)
        if self.config.bidirectional:
            out = out.contiguous().view(-1,2*96)
        else:
            out = out.contiguous().view(-1, 96) 

        out = self.fc(out)
        out = F.sigmoid(out)
        out = out.view(-1, self.rnn_length, parameter.n_classes)
        return out
