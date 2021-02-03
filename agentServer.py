
""" entry for flask service (py 3.6.5)
ver 1.0
Author: Gong Hao, Liu peng, Bingyang li
"""

from flask import *
import jsonpickle
import numpy as np
import time
import sys
import os
import getopt
import json
import ctypes

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


# self defined packages
from agents.policyGradientAgent import*
from agents.ppoAgent import*
from agents.boAgent import*
from agents.deAgent import*
from agents.tsAgent import*
from agents.psoAgent import*

app = Flask(__name__)
websrv_ip = '127.0.0.1'  # default websrv IP for flask
websrv_port = 8080
initFlg=0
agentList=[]
try:
    os.system("mkdir " + "log")
except:
    pass


recordData=[]

rountURLs = ['/autoMLDL/v1.0/initAgent',
             '/autoMLDL/v1.0/getNewParams',
             '/autoMLDL/v1.0/postReward',
            '/autoMLDL/v1.0/doTrain']

nameList = []

@app.route('/')
@app.route('/index')
def index():
    """index page
    """
    with open('README.html', 'r', encoding='utf-8') as file:
        data = file.read()
    return data

@app.route(rountURLs[0], methods=['GET', 'POST'])
def initAgent():
    """show task details in json
    """
    r = request.get_json(force=True)
    print(r)
    agentName=r['UserName']
    agentClass = r['Agent']
    agentExist=False
    agent=None
    reset_flg=False

    for ele in agentList:
        if agentName in ele:
            agentExist=True
            # agent=ele[agentName]

    if "reset" in r:
        reset_flg = r["reset"]

    if agentExist==False:
        if agentClass == 'ppo':
            agent=PPOAgent(r["search_space"],action_features=r["action_features"],num_features=r["num_features"])
        elif agentClass == 'BO':
            agent = boAgent(r["search_space"],init_Xsamples=r["init_Xsamples"],init_Ysamples=r["init_Ysamples"],bigger_is_better = r["bigger_is_better"],
                            Surrogate_function = r["Surrogate_function"],Aquisition_function = r["Aquisition_function"])
        elif agentClass == 'DE':
            agent=deAgent(MIND=r["MIND"],F=r["F"],XOVR=r["XOVR"],SEARCH_SPACE=r["search_space"],bigger_is_better= r["bigger_is_better"])
        elif agentClass == 'PSO':
            agent=psoAgent(MIND=r["MIND"], w=r["w"], c=r["c"], SEARCH_SPACE=r["search_space"], bigger_is_better=r["bigger_is_better"])
        elif agentClass == 'TS':
            agent=tsAgent(tabulen=r["tabulen"], preparelen=r["preparelen"], route=r["route"], bigger_is_better=r["bigger_is_better"])
        agentList.append({agentName:agent})

    elif reset_flg==True:
        if agentClass == 'ppo':
            agent=PPOAgent(r["search_space"],action_features=r["action_features"],num_features=r["num_features"])
        elif agentClass == 'BO':
            agent = boAgent(r["search_space"],init_Xsamples=r["init_Xsamples"],init_Ysamples=r["init_Ysamples"],bigger_is_better = r["bigger_is_better"],
                            Surrogate_function = r["Surrogate_function"],Aquisition_function = r["Aquisition_function"])
        elif agentClass == 'DE':
            agent=deAgent(MIND=r["MIND"],F=r["F"],XOVR=r["XOVR"],SEARCH_SPACE=r["search_space"],bigger_is_better= r["bigger_is_better"])
        elif agentClass == 'PSO':
            agent=psoAgent(MIND=r["MIND"], w=r["w"], c=r["c"], SEARCH_SPACE=r["search_space"], bigger_is_better=r["bigger_is_better"])
        elif agentClass == 'TS':
            agent=tsAgent(tabulen=r["tabulen"], preparelen=r["preparelen"], route=r["route"],bigger_is_better=r["bigger_is_better"])
        
        for ele in agentList:
            if agentName in ele:
                ele[agentName]=agent
    else:
        for ele in agentList:
            if agentName in ele:
                agentExist=True
                agent=ele[agentName]

    actionInfo = agent.makeAction()
    infoData={"info":"initAgent","UserName":agentName,"alreadyExist":agentExist,"currentAgentNum":len(agentList),"reset":reset_flg,"actionInfo":actionInfo}
    
    return json.dumps(infoData)

@app.route(rountURLs[1], methods=['GET', 'POST'])
def getNewParams():
    """
    receive image data and compare it with local faces
    """
    r = request.get_json(force=True)
    print(r)
    agentName=r['UserName']
    agentExist=False
    agent=None
    for ele in agentList:
        if agentName in ele:
            agent=ele[agentName]
            agentExist=True

    if agentExist==False:
        infoData={"info":"getNewParams","warning":"no running agent named "+agentName}
        return json.dumps(infoData)

    if isinstance(agent,PPOAgent):
        clampFlg=r['clampFlg']
        noRepeatFlg=r['noRepeatFlg']
        actionInfo=agent.makeAction(clampFlg=clampFlg,noRepeatFlg=noRepeatFlg,toList=True)
        action_decode,action,action_log_probs,value=actionInfo
    elif isinstance(agent,boAgent):
        actionInfo = agent.makeAction()
    elif isinstance(agent,deAgent):
        actionInfo = agent.makeAction()
    elif isinstance(agent,psoAgent):
        actionInfo = agent.makeAction()
    elif isinstance(agent,tsAgent):
        actionInfo = agent.makeAction()

    infoData={"info":"getNewParams","agentName":agentName,"actionInfo":actionInfo}
    return json.dumps(infoData)

@app.route(rountURLs[2], methods=['GET', 'POST'])
def postReward():
    """verify the permission of current operator
    """
    r = request.get_json(force=True)
    print(r)
    agentName=r['UserName']
    agentExist=False
    agent=None
    for ele in agentList:
        if agentName in ele:
            agent=ele[agentName]
            agentExist=True

    if agentExist==False:
        infoData={"info":"postReward","warning":"no running agent named "+agentName}
        return json.dumps(infoData)

    actionInfo=r["actionInfo"]
    reward=r["reward"]
    if isinstance(agent,PPOAgent):
        agent.addSamples(actionInfo=actionInfo,reward=reward,fromList=True)
        actionInfo = agent.makeAction()
        infoData={"info":"postReward","state":"done","actionInfo":actionInfo}
    elif isinstance(agent,boAgent):
        result_y, result_x = agent.addSamples(actionInfo=actionInfo,reward=reward)
        actionInfo = agent.makeAction()
        infoData={"info":"postReward","state":"done","result_x":result_x,"result_y":result_y,"actionInfo":actionInfo}
    elif isinstance(agent,deAgent):
        agent.addSamples(actionInfo=actionInfo,reward=reward)
        actionInfo = agent.makeAction()
        infoData={"info":"postReward","state":"done","actionInfo":actionInfo}
    elif isinstance(agent,psoAgent):
        agent.addSamples(reward=reward)
        actionInfo = agent.makeAction()
        infoData = {"info":"postReward","state":"done","actionInfo":actionInfo}
    elif isinstance(agent,tsAgent):
        agent.addSamples(reward=reward)
        actionInfo = agent.makeAction()
        infoData = {"info":"postReward","state":"done","actionInfo":actionInfo}

    # logfile= open('log/agentRecord.json','w',encoding='utf-8')
    # recordData.append(r)
    # json.dump(recordData,logfile,ensure_ascii=False)
    # logfile.close()
    
    return json.dumps(infoData)

@app.route(rountURLs[3], methods=['GET', 'POST'])
def doTrain():
    """verify the permission of current operator
    """
    r = request.get_json(force=True)
    print(r)
    agentName=r['UserName']
    agentExist=False
    agent=None
    for ele in agentList:
        if agentName in ele:
            agent=ele[agentName]
            agentExist=True

    if agentExist==False:
        infoData={"info":"doTrain","warning":"no running agent named "+agentName}
        return json.dumps(infoData)

    batchsize=r["batchsize"]
    epoch=r["epoch"]

    agent.offPolicyLearn(epochs=int(epoch),batchsize=int(batchsize))

    infoData={"info":"doTrain","state":"done"}
    return json.dumps(infoData)

def flaskrun(app, default_host, default_port):
    """ Takes a flask.Flask instance and runs it. Parses 
    command-line flags to configure the app.
    """
    import optparse
    print("============================================================================")
    # Set up the command-line options
    parser = optparse.OptionParser()
    parser.add_option("-H", "--host",
                      help="Hostname of the Flask app " +
                      "[default %s]" % default_host,
                      default=default_host)
    parser.add_option("-P", "--port",
                      help="Port for the Flask app " +
                      "[default %s]" % default_port,
                      default=default_port)

    # Two options useful for debugging purposes, but
    # a bit dangerous so not exposed in the help message.
    parser.add_option("-d", "--debug",
                      action="store_true", dest="debug",default="debug",
                      help=optparse.SUPPRESS_HELP)
    parser.add_option("-p", "--profile",
                      action="store_true", dest="profile",
                      help=optparse.SUPPRESS_HELP)
    parser.add_option("-C", "--crtLoc",default="crts", help="Location of cert file")                      

    options, _ = parser.parse_args()

    # If the user selects the profiling option, then we need
    # to do a little extra setup
    if options.profile:
        from werkzeug.contrib.profiler import ProfilerMiddleware

        app.config['PROFILE'] = True
        app.wsgi_app = ProfilerMiddleware(app.wsgi_app,
                                          restrictions=[30])
        options.debug = True
    else:
        options.debug =False
    websrv_ip = options.host
    websrv_port = int(options.port)
    app.run(
        debug=True,
        host=options.host,
        port=int(options.port)
    )  

    strUsage = (
        'Start image worker to calculate image features, usage: ' + ' \n'
        'python run.py' + ' \n'
        '-- end'
    )

if __name__ == '__main__':
    try:
        flaskrun(app, websrv_ip, websrv_port)
    except:
        print('run failed')
        pass