import json
import ctypes
import requests
import sys
import numpy as np
import random


def loadmap(stid=1):
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


# IP = "192.168.1.104"
# IP = "192.168.0.28"
IP = "127.0.0.1"
url_init = "http://" + IP + ":8080/autoMLDL/v1.0/initAgent"
urlAction = "http://" + IP + ":8080/autoMLDL/v1.0/getNewParams"
urlPostReward = "http://" + IP + ":8080/autoMLDL/v1.0/postReward"
infoData = requests.post(url_init, data=json.dumps(data_ts))
# print(infoData.text)
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