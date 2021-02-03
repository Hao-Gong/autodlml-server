import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
#from torch.utils.tensorboard import SummaryWriter
from collections import deque

class RNNACController(nn.Module):
    def __init__(self, decision_features=4, num_steps=4, hidden_size=64, useRandomInit=False):
        super(RNNACController, self).__init__()
        self.useRandomInit=useRandomInit
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
        self.actor= nn.Sequential(
            nn.Linear(hidden_size, decision_features)
            )
        self.critic= nn.Sequential(
            nn.Linear(hidden_size, 1)
            )
        self.num_steps = num_steps
        self.nhid = hidden_size

    def forward(self,batchsize=1):
        actors = []
        critics=[]
        h_t, c_t = self.init_hidden(batchsize)
        input=self.init_input(batchsize)
        for i in range(self.num_steps):
            input=self.base(input)
            h_t, c_t = self.lstm(input, (h_t, c_t))
            actor = self.actor(h_t)
            critic=self.critic(h_t)
            input = actor
            actors += [actor]
            critics += [critic]
        actors = torch.stack(actors).squeeze(1)
        critics = torch.stack(critics).squeeze(1)
        return actors,critics

    def init_input(self,batchsize=1):
        if self.useRandomInit:
            return torch.rand(batchsize, self.decision_features, dtype=torch.float, device=self.DEVICE)
        else:
            return torch.zeros(batchsize, self.decision_features, dtype=torch.float, device=self.DEVICE)

    def init_hidden(self,batchsize=1):
        if self.useRandomInit:
            h_t = torch.rand(batchsize, self.nhid, dtype=torch.float, device=self.DEVICE)
            c_t = torch.rand(batchsize, self.nhid, dtype=torch.float, device=self.DEVICE)
        else:
            h_t = torch.zeros(batchsize, self.nhid, dtype=torch.float, device=self.DEVICE)
            c_t = torch.zeros(batchsize, self.nhid, dtype=torch.float, device=self.DEVICE)
        return (h_t, c_t)