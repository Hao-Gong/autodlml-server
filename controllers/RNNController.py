import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
#from torch.utils.tensorboard import SummaryWriter
from collections import deque

class RNNController(nn.Module):
    def __init__(self, decision_features=4, num_steps=4, hidden_size=64):
        super(RNNController, self).__init__()
        # Could add an embedding layer
        self.DEVICE = torch.device(
                "cuda:0") if torch.cuda.is_available() else 'cpu'
        self.decision_features=decision_features
        self.base=nn.Sequential(
            nn.Linear(decision_features, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        # May be could just use different decoder if these two numbers are the same, not sure
        self.decoder = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, decision_features)
            )
        self.num_steps = num_steps
        self.nhid = hidden_size
        self.hidden = self.init_hidden()

    def forward(self):
        outputs = []
        h_t, c_t = self.hidden
        input=self.init_input()
        for i in range(self.num_steps):
            input=self.base(input)
            h_t, c_t = self.lstm(input, (h_t, c_t))
            output = self.decoder(h_t)
            input = output
            outputs += [output]
        outputs = torch.stack(outputs).squeeze(1)
        return outputs

    def init_input(self):
        # return torch.rand(1, self.decision_features, dtype=torch.float, device=self.DEVICE)
        return torch.zeros(1, self.decision_features, dtype=torch.float, device=self.DEVICE)

    def init_hidden(self):
        # h_t = torch.rand(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        # c_t = torch.rand(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        return (h_t, c_t)