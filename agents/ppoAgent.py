import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
#from torch.utils.tensorboard import SummaryWriter
from collections import deque
from controllers.RNNACController import*
from operator import itemgetter, attrgetter

class PPOAgent(object):
    def __init__(self,action_map, action_features=4, num_steps=4, hidden_size=128, lr=1e-2, useCritic=True):
        super(PPOAgent, self).__init__()
        self.lr=lr
        self.action_features=action_features
        self.num_steps=num_steps
        self.hidden_size=hidden_size
        self.useCritic=useCritic
        self.DEVICE = torch.device(
                "cuda:0") if torch.cuda.is_available() else 'cpu'
        self.controllerModel=RNNACController(action_features, num_steps, hidden_size)
        self.controllerModel.to(self.DEVICE)
        self.optimizer = optim.Adam(params=self.controllerModel.parameters(), lr=self.lr)
        # self.historyRewards = deque([], maxlen=100)
        self.action_map=action_map
        # self.historyQueue=deque([], maxlen=10)
        self.historyQueue=[]
        self.historyQueueLen=20
        # self.historyAction= deque([], maxlen=1000)
        self.historyAction=[]

    def makeAction(self,clampFlg=True, clampEpsilon=0.01,noRepeatFlg=True,toList=False):
        """
            Generate new action 
            Args:
                clampFlg: clamp softmax result, make sure that new action will be made
                noRepeatFlg: check already in history, make sure totally new action will be made
        """
        # generate new action need only 1 batch
        batchsize=1
        # get the action logits from the agent - (preferences)
        # self.controllerModel.eval()
        action_output,value = self.controllerModel(batchsize=batchsize)
        # N*T*S
        action_output=action_output.view(batchsize,self.num_steps,self.action_features)
       #  N*T*1
        value=value.view(batchsize,self.num_steps,1)
        # action_probs.shape N*T*S
        action_probs=softmax(action_output, dim=2)
        # action_distribution.shape N*T*S
        # sample an action according to the action distribution
        if clampFlg:
            action_probs= torch.clamp(action_probs, clampEpsilon, 1.0-clampEpsilon)
            action_distribution=Categorical(probs=action_probs)
            # action.shape N*T*1
            action = action_distribution.sample().unsqueeze(0)
            if noRepeatFlg and (len(self.historyAction)>0 and action.detach().cpu().numpy().tolist() in self.historyAction):
                action = action_distribution.sample().unsqueeze(0)
        else:
            action_distribution=Categorical(probs=action_probs)
            # action.shape N*T*1
            action = action_distribution.sample().unsqueeze(0)
            if noRepeatFlg and (len(self.historyAction)>0 and action.detach().cpu().numpy().tolist() in self.historyAction):
                action = action_distribution.sample().unsqueeze(0)
        # print(action.shape)
        # action_log_probs.shape=N*T*1, log_prob(action) will broadcast the first dim action.
        action_log_probs =action_distribution.log_prob(action)
        # print(action_log_probs.shape)
        action_map_tensor = torch.tensor(self.action_map, device=self.DEVICE)
        action_decode = torch.gather(action_map_tensor , 1, action.view(self.num_steps,1)).squeeze(1)

        self.historyAction.append(action.detach().cpu().numpy().tolist())

        if toList:
            action_decode=action_decode.detach().cpu().numpy().tolist()
            action=action.detach().cpu().numpy().tolist()
            action_log_probs=action_log_probs.detach().cpu().numpy().tolist()
            value=value.detach().cpu().numpy()[:,-1,:].tolist()
            return action_decode,action,action_log_probs,value
        else:
            return action_decode,action,action_log_probs,value
    
    def resampleAction(self,action,batchsize=1):
        # self.controllerModel.train()
        # get the action logits from the agent - (preferences)
        action_output,value = self.controllerModel(batchsize=batchsize)
        # N*T*S
        action_output=action_output.view(batchsize,self.num_steps,self.action_features)
       #  N*T*1
        value=value.view(batchsize,self.num_steps,1)
        # action_probs.shape N*T*S
        action_probs=softmax(action_output, dim=2)
        # print(self.action_probs.shape)
        # action_distribution.shape N*T*S
        action_distribution=Categorical(probs=action_probs)
        # sample an action according to the action distribution
        # action.shape N*T*1
        action_log_probs = action_distribution.log_prob(action)
        # print(action_log_probs.shape)
        return None,None,action_log_probs,value

    def addSamples(self,actionInfo,reward:float,sorted=True,fromList=False):
        _,action,action_log_probs,value=actionInfo
        if fromList==False:
            action=action.detach().cpu().numpy()
            action_log_probs=action_log_probs.detach().cpu().numpy()
            value=value.detach().cpu().numpy()[:,-1,:]

        newRecord={"action":action,"action_log_probs":action_log_probs,"value":value,"reward":reward}
        self.historyQueue.append(newRecord)
        if len(self.historyQueue)>self.historyQueueLen:
            minIndex=0
            minReward=self.historyQueue[0]["reward"]
            # l=[]
            # l.append(self.historyQueue[0]["reward"])
            for i in range(1,len(self.historyQueue)):
                # l.append(self.historyQueue[i]["reward"])
                if(minReward>self.historyQueue[i]["reward"]):
                    minReward=self.historyQueue[i]["reward"]
                    minIndex=i
            del(self.historyQueue[minIndex])

    def offPolicyLearn(self,epochs=1,batchsize=2,clip_epsilon=0.2, kl_beta=0.5,kl_target=0.01, penaty_method="clip"):
        for epoch in range(epochs):
            minibatchIter=self.miniBatchSample(self.historyQueue,batchsize=batchsize)
            for inds in minibatchIter:
                minibatch=self.cat2Batch(inds,self.historyQueue)
                policy_loss,value_loss=self.calculateLoss(minibatch,batchsize,clip_epsilon, kl_beta,kl_target, penaty_method)
                # print(epoch,"/",epochs,policy_loss.item(),value_loss.item())
                self.optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                value_loss.backward(retain_graph=True)
                self.optimizer.step()

    def getHistoryMeanReward(self):
        rewardList=[]
        for ele in self.historyQueue:
            rewardList.append(ele["reward"])
        return np.mean(np.array(rewardList))

    def calculateLoss(self,minibatch,batchsize,clip_epsilon=0.2, kl_beta=0.5,kl_target=0.01, penaty_method="clip"):
        action,action_log_probs,rewards,value=minibatch
        _,_,action_log_probs_resample,value_resample=self.resampleAction(action=action,batchsize=batchsize)
        # only the last value is used
        value_resample=value_resample[:,-1,:]
        # calculate advantage
        if self.useCritic:
            advantages=(rewards-value).view(batchsize,1,-1)
        else:
            # use reward mean
            advantages=(rewards-self.getHistoryMeanReward()).view(batchsize,1,-1)
       # the equation of PI(a|s)/PI_old(a|s)
        pi_ratio=torch.exp(action_log_probs_resample-action_log_probs)
        target1=pi_ratio*advantages
        # print(target1)
        if(penaty_method=="KL"):
            # KL(p||q)=p*log(p/q)
            kl_pen=torch.exp(action_log_probs)*(action_log_probs-action_log_probs_resample)
            # the adaptive kl beta is from the papar of ppo
            if(kl_pen.mean()>4*kl_target): 
                return
            if(kl_pen.mean()<kl_target/1.5):
                kl_beta=kl_target/2
            if(kl_pen.mean()>kl_target*1.5):
                kl_beta=kl_target*2
            # kl
            # the vanilla kl penalty form,  PI(a|s)/PI_old(a|s)*A-beta*KL(PI_old,Pi)
            policy_loss=-(target1-kl_beta*kl_pen).mean()
            # get the values with gradient from the old action
            value_loss=(rewards-value_resample).pow(2).mean()
        # from PPO2 method clip the target
        elif(penaty_method=="clip"):
            target2= torch.clamp(pi_ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
             # maximize the E, so should add -
            policy_loss = -torch.min(target1, target2).mean()
            # get the values with gradient from the old action
            value_loss=(rewards-value_resample).pow(2).mean()
        return policy_loss,value_loss

    def cat2Batch(self,inds,historyQueue):
        action_list=[]
        action_log_probs_list=[]
        reward_list=[]
        value_list=[]
        for i in inds:
            batch=historyQueue[i]
            # print(batch[0])
            action_list.append(torch.tensor(batch["action"]))
            action_log_probs_list.append(torch.tensor(batch["action_log_probs"]))
            value_list.append(torch.tensor(batch["value"]))
            reward_list.append(torch.tensor([batch["reward"]]).view(1,1))

        action=torch.cat(action_list).to(self.DEVICE)
        action_log_probs=torch.cat(action_log_probs_list).to(self.DEVICE)
        reward=torch.cat(reward_list).to(self.DEVICE)
        value=torch.cat(value_list).to(self.DEVICE)

        # print("cat2Batch")
        # print("action",action.shape)
        # print("action_log_probs",action_log_probs.shape)
        # print("reward",reward.shape)
        # print("value",value.shape)
        return action,action_log_probs,reward,value

    #  sampling from the history Queue and return the random sample indexes
    @staticmethod
    def miniBatchSample(historyQueue,batchsize=1):
        inds=np.random.permutation(len(historyQueue))
        batches = inds[:len(inds) // batchsize * batchsize].reshape(-1, batchsize)
        for batch in batches:
            yield batch
        r = len(inds) % batchsize
        if r:
            yield inds[-r:]

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
# agent=PPOAgent(actionMap,action_features=action_features,num_steps=num_steps)

# # targetParam=torch.tensor([5000,5000,5000,5000],dtype=torch.float).cuda()
# # targetParam=torch.tensor([200,200,200,200],dtype=torch.float).cuda()
# targetParam=torch.tensor([50,5000,50,5000],dtype=torch.float).cuda()
# # targetParam=torch.tensor([50,500,500,50],dtype=torch.float).cuda()
# # targetParam=torch.tensor([50,50,50,50],dtype=torch.float).cuda()
# for episode in range(100):
#     for i in range(10):
#         # generate new action
#         actionInfo=agent.makeAction()
#         action_decode,action,action_log_probs,value=actionInfo
#         # calculate reward
#         reward=-torch.pow((action_decode.detach()-targetParam)/5000.0,2).sum().item()
#         # add action batch
#         agent.addSamples(actionInfo=actionInfo,reward=reward)

#     agent.offPolicyLearn(epochs=1,batchsize=1)
#     action_decode,action,action_log_probs,value=agent.makeAction(clampFlg=False,noRepeatFlg=False)
#     reward=-torch.pow((action_decode.detach()-targetParam)/5000.0,2).sum().item()
#     value=value.detach().cpu().numpy()[:,-1,:].tolist()
#     print(action_decode,value,reward)
    