# -*- coding: utf-8 -*-
import numpy as np

class PSOController(object):
    def __init__(self, MIND=100, w=0.6, c=[0.2,0.2], SEARCH_SPACE=None, bigger_is_better=False ):
        self.w = w
        self.c1,self.c2 = c
        self.population_size = MIND
        self.dim = len(SEARCH_SPACE)
        self.x_bound = SEARCH_SPACE
        self.bigger_is_better = bigger_is_better
        self.warmup_flg = True
        self.fitness = np.array([])
        if not self.bigger_is_better:
            self.individual_best_fitness = np.array([float("inf") for _ in range(self.population_size)])
            self.global_best_fitness = float("inf")
        else:
            self.individual_best_fitness = np.array([-float("inf") for _ in range(self.population_size)])
            self.global_best_fitness = -float("inf")
        self.x = np.array([[] for _ in range(self.population_size)])
        for i in range(self.dim):
            self.x = np.concatenate((self.x, np.random.uniform(self.x_bound[i][0], self.x_bound[i][1],
                                       (self.population_size, 1))), axis=1)
        self.mirror = np.array([])
        self.v = np.random.rand(self.population_size, self.dim)
        self.p = self.x.copy()
        self.pg = self.x[0].copy()


    def get_newparam(self):
        print(len(self.mirror))
        if self.warmup_flg:
            if self.mirror.tolist():
                return self.mirror[0]
            else:
                self.warmup_flg = False
                self.mirror = self.x.copy()
                return self.mirror[0]
        else:
            if self.mirror.tolist():
                return self.mirror[0]
            else:
                if not self.bigger_is_better:
                    update_id = np.greater(self.individual_best_fitness, self.fitness)
                else:
                    update_id = np.less(self.individual_best_fitness, self.fitness)

                self.p[update_id] = self.x[update_id]
                self.individual_best_fitness[update_id] = self.fitness[update_id]

                if (not self.bigger_is_better) and (np.min(self.fitness) < self.global_best_fitness):
                    self.pg = self.x[np.argmin(self.fitness)]
                    self.global_best_fitness = np.min(self.fitness)
                elif self.bigger_is_better and (np.max(self.fitness) > self.global_best_fitness):
                    self.pg = self.x[np.argmax(self.fitness)]
                    self.global_best_fitness = np.max(self.fitness)

                self.fitness = np.array([])

                r1 = np.random.rand(self.population_size, self.dim)
                r2 = np.random.rand(self.population_size, self.dim)

                self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + \
                         self.c2 * r2 * (self.pg - self.x)
                self.x = self.v + self.x
                self.mirror = self.x.copy()

                return self.mirror[0]

    def update_reward(self, value):
        if self.warmup_flg:
            self.fitness = np.append(self.fitness, value)
            temp = self.mirror.tolist()
            temp.pop(0)
            self.mirror = np.array(temp)
        else:
            self.fitness = np.append(self.fitness, value)
            temp = self.mirror.tolist()
            temp.pop(0)
            self.mirror = np.array(temp)
