import numpy as np
#from torch import np.random.permutation
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math,time,sys
import random
from datetime import datetime
from copy import deepcopy
import timeit

from config import *

def WOA_mutation(x,it,MaxIteration):
    dim=len(x)
    r=0.9+((-0.9*(it-1))/(MaxIteration-1))
    num_mut=r*dim
    num_mut=int(num_mut)
    changs=[random.randrange(1,dim) for i in range(num_mut)]
    for k in changs:
        x[k]=random.random()
    return x
def crossover(x,y):
    dim=len(x)
    break_point=random.randint(0,dim-1)
    temp=x[break_point:]
    x[break_point:]=y[break_point:]
    y[break_point:]=temp
    return x
    
def WOA_CM(dimension, MaxIter, pop_size, trainX, testX, trainy, testy, pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,round):
    pop_fit=np.copy(pop_fit_init)
    his_best_fit=[]

    best_acc=best_acc_init
    best_cols=np.copy(best_cols_init)
    gbest=np.copy(best_pop_init)
    gbest_fit= best_fit_init
    pop=np.copy(pop_pos_init)
    minn = 1
    maxx = math.floor(0.5*dimension)
    
    if maxx<minn:
        maxx = minn + 1
        #not(c[i].all())
    


    his_best_fit.append(gbest_fit)

    
    for n in range(MaxIter-1):
        print('W Itration : '+str(n)+'-'+str(round)+ '  Fitness: '+str(gbest_fit)+'  Acc: '+str(best_acc)+
                      '  NumF: '+str(len(best_cols))+'  Features: '+str(best_cols))
        a = 2 - 2 * n / (MaxIter - 1)            # linearly decreased from 2 to 0

        for j in range(pop_size):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = np.random.rand()
            b = 1
            if (p < 0.5) :
                if np.abs(A) < 1:
                    #D = np.abs(C * gbest - pop[j] )
                    #pop[j] = gbest - A * D
                    muted=WOA_mutation(gbest,n,MaxIter)
                    pop[j]=crossover(pop[j],muted)
                else :
                    x_rand = pop[np.random.randint(pop_size)] 
                    #D = np.abs(C * x_rand - pop[j])
                    #pop[j] = (x_rand - A * D)
                    
                    pop[j] =crossover(pop[j],x_rand)
            else:
                D1 = np.abs(gbest - pop[j])
                pop[j] = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest 
                               
        for i in range(pop_size):
            for j in range(dimension):
                if (sigmoid(pop[i][j]) > random.random()):
                    pop[i][j] = 1
                else:
                    pop[i][j] = 0
                    
        
        for k in range(pop_size):
            fit,val,cols=Fit_KNN(pop[k], trainX, testX, trainy, testy)
            if fit<gbest_fit:
                gbest=pop[i]
                gbest_fit=fit
                best_acc=val
                best_cols=cols
        his_best_fit.append(gbest_fit)
         
    return  gbest, gbest_fit, his_best_fit,best_acc,best_cols
