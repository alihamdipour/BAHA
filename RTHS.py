import numpy as np
import random
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math,time,sys
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from config import *


Pm = 0.2

def SMO(x):
    for i in range(len(x)):
        random.seed(i**3 + 10 + time.time() ) 
        rnd = random.random()
        if (rnd <= Pm):
            x[i] = 1 - x[i]
        
    return x

def HS(pop,pop_size, fit, dimension, trainX, testX, trainy, testy):    
    
    hybrid = np.array([])
    counter = 0

    for j in range(dimension):
        random.seed(j**3 + 10 + time.time())
        ra = random.randint(0, pop_size-1)
        hybrid = np.append(hybrid, pop[ra][j])

    worst = pop[0]
    for j in range(pop_size):
        if(fit[j] > Fit_KNN(worst, trainX, testX, trainy, testy)[0]):
            worst = deepcopy(pop[j])
            counter = j
    
    tempFit1,tempAcc1,tempCols1=Fit_KNN(worst, trainX, testX, trainy, testy)
    tempFit2,tempAcc2,tempCols2=Fit_KNN(hybrid, trainX, testX, trainy, testy)
    if(tempFit1 > tempFit2):
        fit[counter] = deepcopy(tempFit2)
        pop[counter] = deepcopy(hybrid)
        bestAcc=tempAcc2
        bestCols=tempCols2

    return pop, fit

def RTHS(dimension,MIT,pop_size,trainX, testX, trainy, testy, pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,round):

    his_best_fit=[]
 
    
    bestFit=best_fit_init
    bestCols=list.copy(best_cols_init)
    bestAcc=best_acc_init
    
    pop=pop_pos_init
    fit=np.copy(pop_fit_init)
    his_best_fit.append(bestFit)


   
    for i in range(MIT-1):     
        print('RTHS, itration : '+str(round)+'-'+str(i)+'  Fitness: '+str(bestFit)+'  Acc: '+str(bestAcc)+
            '  NumF: '+str(len(bestCols))+'  Features: '+str(bestCols))   
     
        for j in range(pop_size):
            pop, fit = HS(pop,pop_size, fit, dimension, trainX, testX, trainy, testy)
        for j in range(pop_size):
            random.seed(i**3 + 10 + time.time() )
            one = random.randint(0,pop_size-1)
            two = random.randint(0,pop_size-1)
            three = random.randint(0,pop_size-1)
            four = random.randint(0, pop_size-1)

            One, Two, Three, Four = pop[one], pop[two], pop[three], pop[four]

                           
            y,z = np.array([]), np.array([])
        
            random.seed(i**4 + 40 + time.time()*500)
            r = random.random()
            if (r <= 0.5):
                y = np.append(y, np.add(One, np.multiply(Four, np.add(Two, Three)))%2)
            else:
                y = np.append(y, np.add(One, np.add(Two, Three))%2)

            
            z = np.append(z,SMO(y))
            tempFit1,tempAcc1,tempCols1=Fit_KNN(z, trainX, testX, trainy, testy)
            tempFit2,tempAcc2,tempCols2=Fit_KNN(pop[j], trainX, testX, trainy, testy)
            if(tempFit1 < tempFit2): 
                pop[j] = deepcopy(z)
                fit[j] = deepcopy(tempFit1)
                if fit[j]<bestFit:
                    bestFit=fit[j]
                    bestAcc=tempAcc1
                    bestCols=tempCols1

        his_best_fit.append(bestFit)


    
    return  1,his_best_fit[-1],his_best_fit,bestAcc,bestCols
