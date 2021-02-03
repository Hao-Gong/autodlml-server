from controllers.TabuSearchController import TabuController

class tsAgent():
    def __init__(self, tabulen=50, preparelen=50, route=[], bigger_is_better=False):

        self.optimizer = TabuController(tabulen=tabulen, preparelen=preparelen, route=route.copy(), bigger_is_better=bigger_is_better)

    def makeAction(self):
        return self.optimizer.get_newparam()

    def addSamples(self, reward):
        return self.optimizer.update_reward(reward)