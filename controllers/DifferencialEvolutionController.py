# -*- coding: utf-8 -*-
import numpy as np
import random

def substract(a_list,b_list):
    a = len(a_list)
    new_list = []
    for i in range(0,a):
        new_list.append(a_list[i]-b_list[i])
    return new_list

def add(a_list,b_list):
    a = len(a_list)
    new_list = []
    for i in range(0,a):
        new_list.append(a_list[i]+b_list[i])
    return new_list

def multiply(a,b_list):
    b = len(b_list)
    new_list = []
    for i in range(0,b):
        new_list.append(a * b_list[i])
    return new_list

class DifferentialEvolutionController(object):
    def __init__(self,MIND=100,F=0.6,XOVR=0.7,SEARCH_SPACE=None,bigger_is_better=False):
        self.MIND=MIND
        self.F=F
        self.XOVR=XOVR
        # self.GENERATION=GENERATION
        self.SEARCH_SPACE=SEARCH_SPACE
        self.warmup_flg=True
        self.generation_counter=0
        self.current_generations=[]
        self.sample_generations_regist=[]
        self.new_generations=[]
        self.new_generations_regist=[]
        self.reward_dict={}
        self.init_generation()
        self.bigger_is_better=bigger_is_better
        self.generation_best_record=[]
        self.loss_record=[]

    def init_generation(self):
        for i in range(self.MIND):
            g=self.random_generate()
            self.current_generations.append(g)
            self.sample_generations_regist.append(g)

    def get_newparam(self):
        if(self.warmup_flg):
            if(len(self.sample_generations_regist)>0):
                return self.sample_generations_regist[0]
            else:
                self.warmup_flg=False
                self.record_best(self.reward_dict)
                self.new_generations=self.gen_new_generation(self.current_generations)
                print(self.new_generations)
                self.sample_generations_regist=self.new_generations.copy()
                return self.sample_generations_regist[0]
        else:
            if(len(self.sample_generations_regist)>0):
                return self.sample_generations_regist[0]
            else:
                self.current_generations,self.reward_dict=self.selection(self.new_generations,self.current_generations)
                self.record_best(self.reward_dict)
                self.new_generations=self.gen_new_generation(self.current_generations)
                self.sample_generations_regist=self.new_generations.copy()
                return self.sample_generations_regist[0]

    def record_best(self,m):
        if self.bigger_is_better:
            best_value=max(m.values())
        else:
            best_value=min(m.values())
        for key,value in m.items():
            if(value == best_value):
                self.generation_best_record.append([key,value])
                self.loss_record.append(value)

    def dictkey2list(self,reward_dict):
        l=[]
        for key in reward_dict:
            l.append(list(key))

    def update_reward(self,key,value):
        if(self.warmup_flg):
            self.reward_dict.setdefault(tuple(key), value)
            self.sample_generations_regist.remove(key)
        else:
            self.reward_dict.setdefault(tuple(key), value)
            self.sample_generations_regist.remove(key)

    def gen_new_generation(self,np_list):
        v_list=self.mutation(np_list)
        return self.crossover(np_list,v_list)

    def selection(self,u_list,np_list):
        new_reward_dict={}
        for i in range(0,self.MIND):
            key_u=tuple(u_list[i])
            reward_u=self.reward_dict[key_u]
            key_np=tuple(np_list[i])
            reward_np=self.reward_dict[key_np]
            if self.bigger_is_better:
                if reward_np <= reward_u:
                    np_list[i] = u_list[i]
                    new_reward_dict.setdefault(tuple(key_u), reward_u)
                else:
                    new_reward_dict.setdefault(tuple(key_np), reward_np)
            else:
                if reward_np >= reward_u:
                    np_list[i] = u_list[i]
                    new_reward_dict.setdefault(tuple(key_u), reward_u)
                else:
                    new_reward_dict.setdefault(tuple(key_np),reward_np)
        return np_list,new_reward_dict

    def mutation(self,np_list):
        v_list = []
        for i in range(0,self.MIND):
            r1 = random.randint(0,self.MIND-1)
            while r1 == i:
                r1 = random.randint(0,self.MIND-1)
            r2 = random.randint(0,self.MIND-1)
            while r2 == r1 | r2 == i:
                r2 = random.randint(0,self.MIND-1)
            r3 = random.randint(0,self.MIND-1)
            while r3 == r2 | r3 == r1 | r3 == i:
                r3 = random.randint(0,self.MIND-1)
            v_list.append(add(np_list[r1], multiply(self.F, substract(np_list[r2],np_list[r3]))))
        return v_list

    def crossover(self,np_list,v_list):
        u_list = []
        for i in range(0,self.MIND):
            vv_list = []
            for j in range(0,len(self.SEARCH_SPACE)):
                # if (random.random() <= self.XOVR) | (j == random.randint(0,len(self.SEARCH_SPACE) - 1)):
                if (random.random() <= self.XOVR) :
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])
            u_list.append(vv_list)
        return u_list
    
    def random_generate(self):
        g=[]
        for space in self.SEARCH_SPACE:
            lower=space[0]
            upper=space[1]
            g.append(lower+random.random()*(upper-lower))
        return g

# SEARCH_SPACE=[[-10,10],[-5,5]]
# Traget=[1.1,2.2]

# de=DifferentialEvolutionController(SEARCH_SPACE=SEARCH_SPACE)

# for i in range(10000):) | 