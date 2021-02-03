import random
import matplotlib.pyplot as plt


class TabuController(object):
    def __init__(self, tabulen=50, preparelen=50, route=[], bigger_is_better=False):
        self.tabulen = tabulen
        self.preparelen = preparelen

        self.warmup_flg = True
        self.cost = []
        self.candidate = []
        self.route = route
        self.tabu = []
        self.prepare = []
        self.prepareTabu = []
        self.curroute = self.route.copy()
        self.bigger_is_better = bigger_is_better
        if self.bigger_is_better:
            self.bestcost = -float("inf")
        else:
            self.bestcost = float("inf")
        self.bestroute = self.route

    def randomswap2(self, route):
        route = route.copy()
        while True:
            a = random.choice(route)
            b = random.choice(route)
            if a == b or a == 1 or b == 1:
                continue
            ia = route.index(a)
            ib = route.index(b)
            route[ia] = b
            route[ib] = a
            return route, (a, b)

    def get_newparam(self):
        #print("candidate:", len(self.candidate), "prepare: ",len(self.prepare), "cost: ", len(self.cost))
        if self.warmup_flg:
            if self.candidate:
                return self.candidate[0]
            else:
                self.warmup_flg = False
                rt = self.curroute
                i = 0
                while i < self.preparelen:
                    prt, tabuRec = self.randomswap2(rt)
                    if set(tabuRec) not in self.tabu:
                        self.prepare.append(prt.copy())
                        self.prepareTabu.append(set(tabuRec))
                        i += 1
                self.candidate = self.prepare.copy()
                return self.candidate[0]
        else:
            if self.candidate:
                return self.candidate[0]
            else:
                if not self.bigger_is_better:
                    mc = min(self.cost)
                else:
                    mc = max(self.cost)
                mrt = self.prepare[self.cost.index(mc)]
                mrt_tabu = self.prepareTabu[self.cost.index(mc)]
                if (not self.bigger_is_better) and (mc < self.bestcost):
                    self.bestcost = mc
                    self.bestroute = mrt.copy()
                elif self.bigger_is_better and (mc > self.bestcost):
                    self.bestcost = mc
                    self.bestroute = mrt.copy()
                self.tabu.append(mrt_tabu)
                self.curroute = mrt
                if len(self.tabu) > self.tabulen:
                    self.tabu.pop(0)
                self.prepare, self.prepareTabu, self.cost = [], [], []
                rt = self.curroute
                i = 0
                while i < self.preparelen:
                    prt, tabuRec = self.randomswap2(rt)
                    if set(tabuRec) not in self.tabu:
                        self.prepare.append(prt.copy())
                        self.prepareTabu.append(set(tabuRec))
                        i += 1
                self.candidate = self.prepare.copy()

                return self.candidate[0]

    def update_reward(self, value):
        if self.warmup_flg:
            self.cost.append(value)
            self.candidate.pop(0)
        else:
            self.cost.append(value)
            self.candidate.pop(0)