import numpy as np
from controllers.DifferencialEvolutionController import DifferentialEvolutionController

class deAgent():
    def  __init__(self,MIND=100,F=0.6,XOVR=0.7,SEARCH_SPACE=None,bigger_is_better=False):
        # print(type(MIND))
        # print(type(F))
        # print(type(XOVR))
        self.optimizer = DifferentialEvolutionController(MIND=MIND,F=F,XOVR=XOVR,SEARCH_SPACE=SEARCH_SPACE,bigger_is_better=bigger_is_better)

    def makeAction(self):
        return self.optimizer.get_newparam()
    
    def addSamples(self,actionInfo,reward):
        return self.optimizer.update_reward(actionInfo,reward)

