
## 本工具箱提供多种通用 优化算法 工具：
- 1. [Bayesian Optimization 贝叶斯优化器](#BO)
- 2. [ PPO RL Optimization 近端邻近强化学习优化器](#PPO)
- 3. [DE Optimization 差分进化算法优化器](#DE)
- 4. [PSO Optimization 粒子群优化算法优化器](#PSO)
- 5. [TS Optimization 禁忌搜索算法优化器](#TS)

# Auto DLML服务接口说明

<h1 id="BO">1.Bayesian Optimization 贝叶斯优化器 接口说明</h1>

贝叶斯优化可用来在连续区间内超参数进行搜索优化，一般建议维数不超过20维。

#### 初始化贝叶斯优化器接口

```
(POST) http://+IP:PORT/autoMLDL/v1.0/initAgent
```
参数说明：
```
{
    'UserName':name of your optimizer.
    'Agent':'BO',
    'search_space': bounds of your parameters, should be a list.
    'init_Xsamples': initial samples of parameters,should be a list. 
    'init_Ysamples': Corresponds to init_Xsamples
    'bigger_is_better': whether make the objective function maximum or not,1 corresponds to maximum
    'Surrogate_function': 'GP',
    'Aquisition_function':'EI',
    'reset':False  # 是否重新初始化
}
```
POST之后服务器会返回当前优化器的相关信息

#### 请求下一次采样的参数
```
(POST) http://+IP:PORT/autoMLDL/v1.0/getNewParams
```
参数说明：
```
{
    ‘username':name of optimizer you initialized
}
```
接口以list格式返回相应的下一组搜索的参数值
```
{
    'info':'getNewParams'
    'agentName':name of optimizer you initialized
    'actionInfo':next pair of parameters
}
```


#### 上传所请求参数对应的评价结果
```
(POST) http://+IP:PORT/autoMLDL/v1.0/postReward
```
参数说明：
```
{
    'UserName':name of optimizer you initialized,
    "actionInfo":parameters you get from http://+IP:PORT/autoMLDL/v1.0/getNewParams,
    "reward":value of objective function corresponds to actionInfo}
}
```

#### python 代码案例：

```
import json
import ctypes
import requests
import sys
import numpy as np
###### problem ######
def Hartmann6(x):
    """Hartmann6 function (6-dimensional with 1 global minimum and 6 local minimum)
    minimums = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]
    fmin = -3.32237
    fmax = 0.0 
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    y = 0.0
    for j, alpha_j in enumerate(alpha):
        t = 0
        for k in range(6):
            t += A[j,k] * ((x[k] - P[j,k]) ** 2)
        y -= alpha_j * np.exp(-t)

    return y

def ackley(x, a=20, b=0.2, c=2*np.pi):
    """
    x: vector of input values
    x should be restricted in [-32,32]
    minimums = [0,0,0...]
    fmin = 0
    """
    d = len(x) # dimension of input vector x
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x*x) / d))
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

####### BO test parameters #####
# def objective_function(params):
#     target_param=np.array([1.2,34,5.6])
#     return -np.mean(pow(target_param-np.array(params),2))

def objective_function(params):
    return -ackley(np.array(params))


bounds = np.array([[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15]])
init_x=[]
init_y=[]

for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (10, bounds.shape[0])):
    init_x.append(np.around(params,6).tolist())
    init_y.append(objective_function(np.around(params,6)).tolist())

bound = [[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15]]
data_bo = {'UserName':"Liupeng",'Agent':"BO","search_space":bound,"init_Xsamples":init_x,"init_Ysamples":init_y,
            'bigger_is_better':1,'Surrogate_function': 'GP','Aquisition_function':'EI'}

IP="174.34.106.17"
url_init = "http://"+IP+":8124/autoMLDL/v1.0/initAgent"
urlAction = "http://"+IP+":8124/autoMLDL/v1.0/getNewParams"
urlPostReward="http://"+IP+":8124/autoMLDL/v1.0/postReward"
infoData = requests.post(url_init, data=json.dumps(data_bo),verify=False)

for i in range(300):
    data = {'UserName':"Liupeng"}
    getAction=requests.post(urlAction, data=json.dumps(data),verify=False).json()
    new_x = getAction["actionInfo"]
    reward = float(objective_function(np.array(new_x)))
    #post reward
    data = {'UserName':"Liupeng","actionInfo":new_x,"reward":reward}
    getPostRewardResult=requests.post(urlPostReward, data=json.dumps(data),verify=False).json()
    print(getPostRewardResult)

```


<h1 id="PPO">2.PPO RL Optimization 近端邻近强化学习优化器 接口说明</h1>

算法简介：在具有较少测试样本的情况下可以使用该方式搜索离散状态空间，适合NAS

#### 初始化优化器接口

```
(POST) http://+IP:PORT/autoMLDL/v1.0/initAgent
```
参数说明：
```
{
    'UserName':name of your optimizer,
    "Agent":"ppo",
    'search_space': bounds of your parameters, should be a list,
    "action_features":the descreted number ,
    "num_features":num_features,
    'reset':False  # 是否重新初始化
}
```
POST之后服务器会返回当前优化器的相关信息

#### 请求下一次采样的参数
```
(POST) http://+IP:PORT/autoMLDL/v1.0/getNewParams
```
参数说明：
```
{
    'username':name of optimizer you initialized,
    "noRepeatFlg":True,采样是否能够重复
    "clampFlg":False,概率抽样是否从1截断到0.95
}
```
接口以list格式返回相应的下一组搜索的参数值
```
{
    'info':'getNewParams'
    'agentName':name of optimizer you initialized
    'actionInfo':next pair of parameters
}
```

#### 上传所请求参数对应的评价结果
```
(POST) http://+IP:PORT/autoMLDL/v1.0/postReward
```
参数说明：
```
{
    'UserName':name of optimizer you initialized,
    "actionInfo":parameters you get from http://+IP:PORT/autoMLDL/v1.0/getNewParams,
    "reward":value of objective function corresponds to actionInfo}
}
```

#### python 代码案例：

```

""" entry for flask service (py 3.6.5)
HAO
"""

import json
import ctypes
import requests
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ParamsSetForRL = {
    'SeqModelhidden1':[50,5000],
    'SeqModelhidden2':[50,5000],
    'SeqModelhidden3':[50,5000],
    'SeqModelhidden4':[50,5000],
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

IP="174.34.106.17"
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

```


<h1 id="DE">3. DE Optimization 差分进化算法优化器 接口说明</h1>

算法简介：差分进化算法是目前较强的全局寻优算法，在能够快速评价的时候可以使用

#### 初始化优化器接口

```
(POST) http://+IP:PORT/autoMLDL/v1.0/initAgent
```
参数说明：
```
{
    'UserName':name of your optimizer,
    "Agent":"DE",
    "search_space":bound,
    "bigger_is_better":False,
    "MIND": 种群数量,
    "F":0.6,变异概率
    "XOVR":0.7,交叉概率
    "reset":False
}
```
POST之后服务器会返回当前优化器的相关信息

#### 请求下一次采样的参数
```
(POST) http://+IP:PORT/autoMLDL/v1.0/getNewParams
```
参数说明：
```
{
    ‘username':name of optimizer you initialized
}
```
接口以list格式返回相应的下一组搜索的参数值
```
{
    'info':'getNewParams'
    'agentName':name of optimizer you initialized
    'actionInfo':next pair of parameters
}
```

#### 上传所请求参数对应的评价结果
```
(POST) http://+IP:PORT/autoMLDL/v1.0/postReward
```
参数说明：
```
{
    'UserName':name of optimizer you initialized,
    "actionInfo":parameters you get from http://+IP:PORT/autoMLDL/v1.0/getNewParams,
    "reward":value of objective function corresponds to actionInfo}
}
```

#### python 代码案例：

```
""" entry for flask service (py 3.6.5)
HAO
"""

import json
import ctypes
import requests
import sys
import numpy as np

###### problem ######
def Hartmann6(x):
    """Hartmann6 function (6-dimensional with 1 global minimum and 6 local minimum)
    minimums = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]
    fmin = -3.32237
    fmax = 0.0 
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    y = 0.0
    for j, alpha_j in enumerate(alpha):
        t = 0
        for k in range(6):
            t += A[j,k] * ((x[k] - P[j,k]) ** 2)
        y -= alpha_j * np.exp(-t)

    return y

def ackley(x, a=20, b=0.2, c=2*np.pi):
    """
    x: vector of input values
    x should be restricted in [-32,32]
    minimums = [0,0,0...]
    fmin = 0
    """
    d = len(x) # dimension of input vector x
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x*x) / d))
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term


def objective_function(params):
    return -Hartmann6(np.array(params))


bounds = np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
init_x=[]
init_y=[]

for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (10, bounds.shape[0])):
    init_x.append(np.around(params,6).tolist())
    init_y.append(objective_function(np.around(params,6)).tolist())

bound = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
# deAgent(MIND=["MIND"],F=["F"],XOVR=r["XOVR"],SEARCH_SPACE=r["search_space"],bigger_is_better=bigger_is_better = r["bigger_is_better"])
data_de = {'UserName':"Gonghao","Agent":"DE","search_space":bound,"bigger_is_better":False,"MIND": 100,"F":0.6,"XOVR":0.7,"reset":True}

IP="174.34.106.17"
url_init = "http://"+IP+":8124/autoMLDL/v1.0/initAgent"
urlAction = "http://"+IP+":8124/autoMLDL/v1.0/getNewParams"
urlPostReward="http://"+IP+":8124/autoMLDL/v1.0/postReward"
infoData = requests.post(url_init, data=json.dumps(data_de))
print(infoData.text)
for i in range(10000):
    data = {'UserName':"Gonghao"}
    getAction=requests.post(urlAction, data=json.dumps(data)).json()
    # print(getAction)
    new_x = getAction["actionInfo"]
    
    reward = float(objective_function(np.array(new_x)))
    print(new_x,reward)
    #post reward
    data = {'UserName':"Gonghao","actionInfo":new_x,"reward":reward}
    getPostRewardResult=requests.post(urlPostReward, data=json.dumps(data)).json()

```

<h1 id="PSO">4. PSO Optimization 粒子群优化算法优化器 接口说明</h1>

算法简介：粒子全算法(Particle swarm optimization)是一种常用的基于种群的全局优化算法，搜索速度快，具有记忆性，被广泛地用于各种函数优化问题中。

#### 初始化优化器接口

```
(POST) http://+IP:PORT/autoMLDL/v1.0/initAgent
```
参数说明：
```
{
    "UserName": name of your optimizer,
    "Agent": "PSO",
    "search_space": bound,
    "bigger_is_better": False (when handling minimum optimization) or True (when handling maximum optimization),
    "MIND": 50(default), the population size,
    "w": 0.6(default), the inertia factor, used to adjust the balance degree between global and local optimization ability. 
    "c": [0.2, 0.2], accleration constant, used to adjust the weight of learning ability of individual and social learning experience.
    "reset": False
}
```
POST之后服务器会返回当前优化器的相关信息

#### 请求下一次采样的参数
```
(POST) http://+IP:PORT/autoMLDL/v1.0/getNewParams
```
参数说明：
```
{
    "username": name of your intialized optimizer
}
```
接口以list格式返回相应的下一组搜索的参数值
```
{
    "info": "getNewParams",
    "agentName": name of your initialized optimizer,
    "actionInfo": info of the next solution in the generated population  
}
```

#### 上传所请求参数对应的评价结果
```
(POST) http://+IP:PORT/autoMLDL/v1.0/postReward
```
参数说明：
```
{
    "UserName": name of your initialized optimizer,
    "actionInfo": the results you request from http://+IP:PORT/autoMLDL/v1.0/getNewParams,
    "reward": the objective value of results in the last step,
}
```

#### python 代码案例：

```
import json
import ctypes
import requests
import sys
import numpy as np

def Hartmann6(x):
    """Hartmann6 function (6-dimensional with 1 global minimum and 6 local minimum)
    minimums = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]
    fmin = -3.32237
    fmax = 0.0 
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    y = 0.0
    for j, alpha_j in enumerate(alpha):
        t = 0
        for k in range(6):
            t += A[j,k] * ((x[k] - P[j,k]) ** 2)
        y -= alpha_j * np.exp(-t)

    return y

def objective_function(params):
""" SAME AS THE EXAMPLE LISTED IN DE AGENT"""
    return -Hartmann6(np.array(params))

bound = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]

data_pso = {"UserName": "Li by", "Agent": "PSO", "search_space": bound, "bigger_is_better": False, "MIND": 50, "w": 0.6, "c": [0.2,0.2], "reset": True}

IP = "174.34.106.17"
url_init = "http://" + IP + ":8124/autoMLDL/v1.0/initAgent"
urlAction = "http://" + IP + ":8124/autoMLDL/v1.0/getNewParams"
urlPostReward = "http://" + IP + ":8124/autoMLDL/v1.0/postReward"

infoData = requests.post(url_init, data=json.dumps(data_pso)).json()
print(infoData)
# print(infoData.text)
for i in range(1000):
    if not i%data_pso[MIND]:
        data = {"UserName": "Li by"}
        getAction = requests.post(urlAction, data=json.dumps(data)).json()
        print(getAction)
        new_x = getAction["actionInfo"]

        reward = objective_function(np.array(new_x))
        print(i, "new_x: ", new_x, "reward: ", reward, "\n")
        data = {"UserName": "Li by", "actionInfo": new_x, "reward": reward}
        getPostRewardResult = requests.post(urlPostReward, data=json.dumps(data)).json()

```

<h1 id="TS">5. TS Optimization 禁忌搜索算法优化器 接口说明</h1>

算法简介：禁忌搜索(Tabu search)是一种有效的用来跳出局部最优解的启发式搜索算法，在组合优化、路径规划以及生产调度问题中具有非常好的表现。

#### 初始优化器接口

```
(POST) http://+IP:PORT/autoMLDL/v1.0/initAgent
```
参数说明：
```
{
    "UserName": name of your optimizer,
    "Agent": "TS",
    "search_space": bound,
    "bigger_is_better": False (when handling minimum optimization) or True (when handling maximum optimization),
    "tabulen": 50(default), the length of tabu list,
    "preparelen": 50(default), the prepared candidate set for the current solution. 
    "route": to random generate one legal initial solution, you can define it case by case.
    "reset": False
}
```
POST之后服务器会返回当前优化器的相关信息

#### 请求下一次采样的参数
```
(POST) http://+IP:PORT/autoMLDL/v1.0/getNewParams
```
参数说明：
```
{
    ‘username':name of optimizer you initialized
}
```
接口以list格式返回相应的下一组搜索的参数值
```
{
    "info": "getNewParams",
    "agentName": name of your initialized optimizer,
    "actionInfo": info of the next solution ready to be computed  
}
```

#### 上传所请求参数对应的评价结果
```
(POST) http://+IP:PORT/autoMLDL/v1.0/postReward
```
参数说明：
```
{
    "UserName": name of your initialized optimizer,
    "actionInfo": the results you request from http://+IP:PORT/autoMLDL/v1.0/getNewParams,
    "reward": the objective value of results in the last step,
}
```
#### python 代码案例：

```

import json
import ctypes
import requests
import sys
import numpy as np
import random


def loadmap(stid=1):

""" In this case, the traveling salesman problem is used as the illustration example, 
and the following is the info of map, where the key represents ID of different citys and 
the numbers in tuple represent the longitude and latitude info of each city"""

    map = {1: (1150.0, 1760.0), 2: (630.0, 1660.0), 3: (40.0, 2090.0), 4: (750.0, 1100.0),
            5: (750.0, 2030.0), 6: (1030.0, 2070.0), 7: (1650.0, 650.0), 8: (1490.0, 1630.0),
            9: (790.0, 2260.0), 10: (710.0, 1310.0), 11: (840.0, 550.0), 12: (1170.0, 2300.0),
            13: (970.0, 1340.0), 14: (510.0, 700.0), 15: (750.0, 900.0), 16: (1280.0, 1200.0),
            17: (230.0, 590.0), 18: (460.0, 860.0), 19: (1040.0, 950.0), 20: (590.0, 1390.0),
            21: (830.0, 1770.0), 22: (490.0, 500.0), 23: (1840.0, 1240.0), 24: (1260.0, 1500.0),
            25: (1280.0, 790.0), 26: (490.0, 2130.0), 27: (1460.0, 1420.0), 28: (1260.0, 1910.0),
            29: (360.0, 1980.0)}
    mapid = list(map.keys())
    return map, mapid, stid

def randomroute(stid, mapids):

    stid = stid
    rt = mapids.copy()
    random.shuffle(rt)
    rt.pop(rt.index(stid))
    rt.insert(0, stid)
    return rt

def objective_function(map, road):

    d = -1
    st = 0, 0
    cur = 0, 0
    map = map
    for v in road:
        if d == -1:
            st = map[v]
            cur = st
            d = 0
        else:
            d += ((cur[0] - map[v][0]) ** 2 + (cur[1] - map[v][1]) ** 2) ** 0.5
            cur = map[v]
    d += ((cur[0] - st[0]) ** 2 + (cur[1] - st[1]) ** 2) ** 0.5
    return d

map, mapid, stid = loadmap(1)
route = randomroute(stid, mapid)


data_ts = {'UserName': "Li by", "Agent": "TS", "bigger_is_better": False, "tabulen":50, "preparelen":50, "route":route, "reset": True}

IP = "174.34.106.17"
url_init = "http://" + IP + ":8080/autoMLDL/v1.0/initAgent"
urlAction = "http://" + IP + ":8080/autoMLDL/v1.0/getNewParams"
urlPostReward = "http://" + IP + ":8080/autoMLDL/v1.0/postReward"
infoData = requests.post(url_init, data=json.dumps(data_ts))
for i in range(5000):
    data = {'UserName': "Li by"}
    getAction = requests.post(urlAction, data=json.dumps(data)).json()
    print(getAction)
    new_x = getAction["actionInfo"]
    #print(new_x)

    reward = float(objective_function(map, new_x))
    print(i, "new_x: ", new_x, "reward: ", reward, "\n")
    # post reward
    data = {'UserName': "Li by", "actionInfo": new_x, "reward": reward}
    getPostRewardResult = requests.post(urlPostReward, data=json.dumps(data)).json()

```