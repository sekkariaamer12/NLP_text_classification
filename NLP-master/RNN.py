import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, emb_size, device):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.i2e = nn.Linear(input_size, emb_size, device=self.device)
        self.i2h = nn.Linear(emb_size + hidden_size, hidden_size, device=self.device)
        self.i2o = nn.Linear(emb_size + hidden_size, output_size, device=self.device)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.i2e(input)
        combined = torch.cat((embedded, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(128, self.hidden_size, device=self.device)
