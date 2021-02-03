
""" entry for flask service (py 3.6.5)
HAO UAES.AI
"""

import json
import ctypes
import requests
import sys

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
from policyGradientAgent import*
from ppoAgent import*


ParamsSetForRL = {
    # 'small_var_th':[0.0,0.1], 
    # 'high_corr_th':[0.9,0.95], 
    # 'low_toY_th':[0.01,0.02],
    # DL params start
    # 'lr_inDL':[1e-5,1e-2],
    'SeqModelhidden1':[50,5000],
    'SeqModelhidden2':[50,5000],
    'SeqModelhidden3':[50,5000],
    'SeqModelhidden4':[50,5000],
    # 'rnn_hidden_size_inDL':[50,100,200], 
    # 'epoches':[100]
    # DL params end
}
searchGrid=10

targetParam=np.array([50,5000,50,5000])

def paramMinMax2Map(params,grid=10):
    map=[]
    for m in params:
        m_grid=np.linspace(m[0],m[1],grid).tolist()
        map.append(m_grid)
    return map

print(ParamsSetForRL.values())
search_space=paramMinMax2Map(ParamsSetForRL.values(),grid=searchGrid)
action_features=searchGrid
num_features=len(search_space)

IP="0.0.0.0"
url = "http://"+IP+":8124/autoMLDL/v1.0/initAgent"
data = {'UserName':"Gonghao",'Agent':"ppo","search_space":search_space,"action_features":action_features,"num_features":num_features}
print(requests.post(url, data=json.dumps(data),verify=False).text)


urlAction = "http://"+IP+":8124/autoMLDL/v1.0/getNewParams"
urlPostReward="http://"+IP+":8124/autoMLDL/v1.0/postReward"
urlDoTrain="http://"+IP+":8124/autoMLDL/v1.0/doTrain"

for epoch in range(100):
    for i in range(10):
        # get new action
        data = {'UserName':"Gonghao","noRepeatFlg":True,"clampFlg":False}
        getAction=requests.post(urlAction, data=json.dumps(data),verify=False).json()
        # print(getAction)
        actionInfo=getAction["actionInfo"]
        action_decode,action,action_log_probs,value=actionInfo
        # calculate reward
        action_decode=np.array(action_decode,float)
        reward=-sum(pow((action_decode-targetParam)/5000.0,2))
        # print("action:",action_decode,"  running reward:",reward)

        # post reward
        data = {'UserName':"Gonghao","actionInfo":actionInfo,"reward":reward}
        getPostRewardResult=requests.post(urlPostReward, data=json.dumps(data),verify=False).json()
        # print(getPostRewardResult)

    data= {'UserName':"Gonghao","batchsize":1,"epoch":1}
    getDoTrainResult=requests.post(urlDoTrain, data=json.dumps(data),verify=False).json()
    # print(getDoTrainResult)
    # Test
    data = {'UserName':"Gonghao","noRepeatFlg":False,"clampFlg":False}
    getAction=requests.post(urlAction, data=json.dumps(data),verify=False).json()
    # print(getAction)
    actionInfo=getAction["actionInfo"]
    action_decode,action,action_log_probs,value=actionInfo
    # calculate reward
    action_decode=np.array(action_decode,float)
    reward=-sum(pow((action_decode-targetParam)/5000.0,2))
    print("action:",action_decode,"  running reward:",reward)

