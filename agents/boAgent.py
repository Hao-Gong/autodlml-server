import numpy as np
from controllers.BayesianOptController import*

class boAgent():
    def __init__(self,action_map,init_Xsamples,init_Ysamples,bigger_is_better =1,
                    Surrogate_function = 'GP',Aquisition_function = 'EI'):
        self.bound = np.array(action_map)
        self.init_Xsamples = init_Xsamples
        self.init_Ysamples = init_Ysamples        
        self.optimizer = BayesianOptController(self.bound, self.init_Xsamples,self.init_Ysamples,bigger_is_better,
                                                Surrogate_function,Aquisition_function)
        # print('initial boAgent')
    def makeAction(self):
        next_sample = self.optimizer.next_hyperparams()
        next_sample = next_sample.tolist()
        print('get next sample:',next_sample[0])
        return next_sample[0]
    
    def addSamples(self,actionInfo,reward):
        return self.optimizer.update(actionInfo,reward)
