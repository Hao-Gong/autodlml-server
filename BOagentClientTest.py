import json
import ctypes
import requests
import sys
import numpy as np
from OptimizationTestProblems import Hartmann6,ackley

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

IP="0.0.0.0"
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

