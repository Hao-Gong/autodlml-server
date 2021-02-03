# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from numpy.linalg import cholesky, det, lstsq


class BayesianOptController():
    def __init__(self, bound,init_Xsamples,init_Ysamples,bigger_is_better,Surrogate_function,Aquisition_function):
        self.bounds_range = bound
        self.x_sample = init_Xsamples
        self.X = np.array(self.x_sample)
        self.y_sample = init_Ysamples
        self.Y = np.array(self.y_sample).reshape(-1,1)
        self.bigger_is_better = bigger_is_better

        if Surrogate_function == 'GP':
            self.surrogate =GPR()
            self.surrogate.fit(self.X,self.Y)
        self.acquisition = Aquisition_function
        self.acq =aquisition()
        self.xi = 0.01
    
    def mini_obj(self,x):
        return -self.acq.run(self.surrogate,x,self.X,self.Y,self.acquisition,self.bigger_is_better,self.xi)
    
    def next_hyperparams(self,method='L-BFGS-B',n_restart=25):
        best_aq =1
        global next_x
        for params in np.random.uniform(self.bounds_range[:, 0], self.bounds_range[:, 1], (n_restart, self.bounds_range.shape[0])):
            res = minimize(fun=self.mini_obj,x0 = params.reshape(1,-1),bounds = self.bounds_range,method =method)
            if res.fun < best_aq:
                best_aq = res.fun[0]
                next_x = res.x
        return next_x.reshape(1,-1)
    
    def update(self,new_x,new_y):
        self.x_sample.append(new_x)
        self.X = np.array(self.x_sample)
        self.y_sample.append(new_y)
        self.Y = np.array(self.y_sample)
        print('update x _sample:',self.X)
        print('update y _sample:',self.Y)
        self.surrogate.fit(self.X,self.Y)
        
        return np.max(self.Y).tolist(),self.X[np.argmax(self.Y)].tolist()


class aquisition: 
    def __init__(self): 
        pass
        
    def expected_improvement(self,gpr, X, X_sample, Y_sample, bigger_is_better, xi=0.01):
        mu,cov = gpr.predict(X)
        sigma = np.diag(cov).reshape(-1,1)
        maxmin_flag = (-1)** (not bigger_is_better) 
        if bigger_is_better:
            opt_LastRun = np.max(Y_sample)
        else:
            opt_LastRun = np.min(Y_sample)
        with np.errstate(divide='warn'):           
            temp = (mu - opt_LastRun - xi).reshape(-1,1)
            z=temp/sigma
            ei = temp*norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0.0] = 0.0 
        return ei
    
    def run(self,gpr, X, X_sample, Y_sample, aquisition_name, bigger_is_better, xi=0.01):
        if aquisition_name =='EI':
            return self.expected_improvement(gpr, X, X_sample, Y_sample, bigger_is_better, xi=0.01)

class GPR:
    
    def __init__(self,l = 1.0, sigma = 1.0,optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": l, "sigma_f": sigma}
        self.optimize = optimize
        
    def fit(self,x,y):
        self.train_X = np.asarray(x)
        self.train_y = np.asarray(y)
        self.mean_y = np.mean(self.train_y)
        self.train_y = self.train_y-self.mean_y # regularize y
         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            global kyy_inv
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            try:
                kyy_inv = np.linalg.inv(Kyy)
            except:
                print("No inv matrix for kyy")
            return 0.5 * self.train_y.T.dot(kyy_inv).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                   bounds=((1e-4, 1e4), (1e-4, 1e4)),
                   method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]
        self.is_fit = True         
        
    def kernel(self,x1,x2):
        if len(x1.shape)==1:
            X1 = x1.reshape(1,-1)
        else:
            X1=x1            
        if len(x2.shape)==1:
            X2 = x2.reshape(1,-1)
        else:
            X2=x2
        dist_matrix = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix.astype(float))
    
    def predict(self,x):
        X = np.asarray(x)
        global kff_inv
        if not self.is_fit:
            print('U should fit first')
            return
        kff = self.kernel(self.train_X,self.train_X)+ 1e-8 * np.eye(len(self.train_X))
        kyy = self.kernel(x,x)
        kfy = self.kernel(self.train_X,x)
        try:
            kff_inv = np.linalg.inv(kff)
        except:
            print("No inv matrix for kff")
        #average
        miu = kfy.T.dot(kff_inv).dot(self.train_y)+self.mean_y # add regularizition value
        cov = kyy - kfy.T.dot(kff_inv).dot(kfy)
        return miu,cov 