import json
import ctypes
import requests
import sys
import numpy as np
from OptimizationTestProblems import Hartmann6

####### BO test parameters #####
# def objective_function(params):
#     target_param=np.array([1.2,34,5.6])
#     return -np.mean(pow(target_param-np.array(params),2))

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

IP="0.0.0.0"
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
    # print(int(reward))

