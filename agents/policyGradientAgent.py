import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
#from torch.utils.tensorboard import SummaryWriter
from collections import deque
from controllers.RNNController import*

class PolicyGradient(object):
    def __init__(self,action_map, action_features=4, num_steps=4, hidden_size=32, lr=2e-3,beta=0.1,gamma=0.99):
        self.lr=lr
        self.action_features=action_features
        self.num_steps=num_steps
        self.hidden_size=hidden_size
        self.beta=beta
        self.gamma=gamma
        self.DEVICE = torch.device(
                "cuda:0") if torch.cuda.is_available() else 'cpu'
        self.controllerModel=RNNController(action_features, num_steps, hidden_size)
        self.controllerModel.to(self.DEVICE)
        self.optimizer = optim.Adam(params=self.controllerModel.parameters(), lr=self.lr)
        self.total_rewards = deque([], maxlen=100)
        self.action_map=action_map

    def makeAction(self,batchsize=1):

        # get the action logits from the agent - (preferences)
        # self.controllerModel.eval()
        # self.action_output.shape N*T*S
        self.action_output= self.controllerModel().view(batchsize,self.num_steps,self.action_features)
        # print(self.action_output.shape)
        # action_probs.shape N*T*S
        self.action_probs=softmax(self.action_output, dim=2)
        # print(self.action_probs.shape)
        # action_distribution.shape N*T*S
        action_distribution=Categorical(probs=self.action_probs)
        # sample an action according to the action distribution
        # action.shape N*T*1
        action = action_distribution.sample().unsqueeze(0)
        # print(action.shape)
        # action_log_probs.shape=N*T*1, log_prob(action) will broadcast the first dim action.
        self.action_log_probs =action_distribution.log_prob(action)
        # print(self.action_log_probs.shape)
        action_map_tensor = torch.tensor(self.action_map, device=self.DEVICE)
        action_decode = torch.gather(action_map_tensor , 1, action.view(self.num_steps,1)).squeeze(1)
        return action_decode

    def learn(self,reward):
        self.total_rewards.append(reward)
        weighted_log_probs  = self.action_log_probs * (reward-np.mean(self.total_rewards))
        # after each episode append the sum of total rewards to the deque
        
        # calculate the loss
        loss, entropy = self.calculate_loss(epoch_logits=self.action_output,
                                            weighted_log_probs=weighted_log_probs)
        # zero the gradient
        self.optimizer.zero_grad()
        # backprop
        loss.backward()
        # update the parameters
        self.optimizer.step()
        # feedback
        print("\r", f" Avg Return per Epoch: {np.mean(self.total_rewards):.3f}")

    def calculate_loss(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor, has_entropy_bonus=True) -> (torch.Tensor, torch.Tensor):
        """
            Calculates the policy "loss" and the entropy bonus
            Args:
                epoch_logits: logits of the policy network we have collected over the epoch
                weighted_log_probs: loP * W of the actions taken
            Returns:
                policy loss + the entropy bonus
                entropy: needed for logging
        """
        policy_loss = -torch.mean(weighted_log_probs)

        # add the entropy bonus
        if(has_entropy_bonus):
            p = softmax(epoch_logits, dim=2)
            log_p = log_softmax(epoch_logits, dim=2)
            entropy = -torch.mean(torch.sum(p * log_p), dim=0)
            entropy_bonus = -self.beta* entropy

            return policy_loss+ entropy_bonus, entropy
        else:
            return policy_loss, entropy


# ParamsSetForRL = {
#     # 'small_var_th':[0.0,0.1], 
#     # 'high_corr_th':[0.9,0.95], 
#     # 'low_toY_th':[0.01,0.02],
#     # DL params start
#     # 'lr_inDL':[1e-5,1e-2],

#     'SeqModelhidden1':[50,5000],
#     'SeqModelhidden2':[50,5000],
#     'SeqModelhidden3':[50,5000],
#     'SeqModelhidden4':[50,5000],
#     # 'rnn_hidden_size_inDL':[50,100,200], 
#     # 'epoches':[100]
#     # DL params end
# }
# searchGrid=10

# def paramMinMax2Map(params,grid=10):
#     map=[]
#     for m in params:
#         m_grid=np.linspace(m[0],m[1],grid).tolist()
#         map.append(m_grid)
#     return map

# print(ParamsSetForRL.values())
# actionMap=paramMinMax2Map(ParamsSetForRL.values(),grid=searchGrid)
# action_features=searchGrid
# num_steps=len(actionMap)
# PGAgent=PolicyGradient(actionMap,action_features=action_features,num_steps=num_steps)

# # targetParam=torch.tensor([5000,5000,5000,5000],dtype=torch.float).cuda()
# # targetParam=torch.tensor([1000,1000,1000,1000],dtype=torch.float).cuda()
# targetParam=torch.tensor([50,5000,50,5000],dtype=torch.float).cuda()
# # targetParam=torch.tensor([50,500,500,50],dtype=torch.float).cuda()
# # targetParam=torch.tensor([50,50,50,50],dtype=torch.float).cuda()
# for i in range(1000):
#     action=PGAgent.makeAction()
#     print(action)
#     # reward=-max(abs(action-targetParam)).item()
#     # reward=-torch.pow((action.detach()-targetParam),2).sum().item()
#     reward=-torch.pow((action.detach()-targetParam),2).sum().item()
#     print(targetParam)
#     print(reward)
#     PGAgent.learn(reward)