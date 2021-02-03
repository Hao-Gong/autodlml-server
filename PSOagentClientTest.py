import json
import ctypes
import requests
import sys
import numpy as np
from OptimizationTestProblems import Hartmann6

def objective_function(params):
    return -Hartmann6(np.array(params))

# def objective_function( x):
#     return np.sum(np.square(x)).tolist()


# bound = [[-10,10]]
bound = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]

data_pso = {"UserName": "Li by", "Agent": "PSO", "search_space": bound, "bigger_is_better": False, "MIND": 50, "w": 0.6, "c": [0.2,0.2], "reset": True}

# IP = "192.168.1.104"
# IP = "192.168.0.28"
IP = "127.0.0.1"
url_init = "http://" + IP + ":8080/autoMLDL/v1.0/initAgent"
urlAction = "http://" + IP + ":8080/autoMLDL/v1.0/getNewParams"
urlPostReward = "http://" + IP + ":8080/autoMLDL/v1.0/postReward"

infoData = requests.post(url_init, data=json.dumps(data_pso)).json()
print(infoData)
# print(infoData.text)
for i in range(1000):
    data = {"UserName": "Li by"}
    getAction = requests.post(urlAction, data=json.dumps(data)).json()
    print(getAction)
    new_x = getAction["actionInfo"]

    reward = objective_function(np.array(new_x))
    print(i, "new_x: ", new_x, "reward: ", reward, "\n")
    # post reward
    data = {"UserName": "Li by", "actionInfo": new_x, "reward": reward}
    getPostRewardResult = requests.post(urlPostReward, data=json.dumps(data)).json()