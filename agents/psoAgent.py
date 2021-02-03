from controllers.ParticleSwarmOptimizationController import PSOController

class psoAgent():
    def __init__(self, MIND=100, w=0.6, c=None, SEARCH_SPACE=None, bigger_is_better=False):

        self.optimizer = PSOController(MIND=MIND, w=w, c=c, SEARCH_SPACE=SEARCH_SPACE, bigger_is_better=bigger_is_better)

    def makeAction(self):
        return self.optimizer.get_newparam().tolist()

    def addSamples(self, reward):
        return self.optimizer.update_reward(reward)