import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math,time,sys
import random
from datetime import datetime
from copy import deepcopy
from config import *



def onecnt(agent):
    return sum(agent)

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))
    
def fraction(pop_fit,pop_pfit):
    k = 0
    for i in range(len(pop_fit)):
        if(pop_fit[i] > pop_pfit[i]):
            k = k+1
            
    return k

def GWO(pop, alpha, beta, delta, pop_size, dimension, a):
    for i in range(pop_size):
        for j in range (dimension):     

            r1=random.random() # r1 is a random number in [0,1]
            r2=random.random() # r2 is a random number in [0,1]

            A1=2*a*r1-a; # Equation (3.3)
            C1=2*r2; # Equation (3.4)

            D_alpha=abs(C1*alpha[j]-pop[i,j]); # Equation (3.5)-part 1
            X1=alpha[j]-A1*D_alpha; # Equation (3.6)-part 1

            r1=random.random()
            r2=random.random()

            A2=2*a*r1-a; # Equation (3.3)
            C2=2*r2; # Equation (3.4)

            D_beta=abs(C2*beta[j]-pop[i,j]); # Equation (3.5)-part 2
            X2=beta[j]-A2*D_beta; # Equation (3.6)-part 2       

            r1=random.random()
            r2=random.random() 

            A3=2*a*r1-a; # Equation (3.3)
            C3=2*r2; # Equation (3.4)

            D_delta=abs(C3*delta[j]-pop[i,j]); # Equation (3.5)-part 3
            X3=delta[j]-A3*D_delta; # Equation (3.5)-part 3             

            pop[i,j]=(X1+X2+X3)/3  # Equation (3.7)
            
    return pop

def WOA(pop, pop_size, alpha, a):
    for j in range(pop_size):

        r = np.random.rand()
        A = 2 * a * r - a
        C = 2 * r
        l = np.random.uniform(-1, 1)
        p = np.random.rand()
        b = 1

        if (p < 0.5) :
            if np.abs(A) < 1:
                D = np.abs(C * alpha - pop[j] )
                pop[j] = alpha - A * D
            else :
                x_rand = pop[np.random.randint(pop_size)] 
                D = np.abs(C * x_rand - pop[j])
                pop[j] = (x_rand - A * D)
        else:
            D1 = np.abs(alpha - pop[j])
            pop[j] = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + alpha

    return pop

    
def ASGW(dimension, MaxIter, pop_size, trainX, testX, trainy, testy,  pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,round):
    his_best_fit=[]    


    #temp_cols = np.empty(pop_size, dtype=object)
    corspod_best_acc = best_acc_init
    corspod_best_cols=best_cols_init
    corspod_best_fit=best_fit_init
    new_fit=float('inf')
    new_acc=float('inf')
    new_cols=[]

    pop=pop_pos_init

    fit = pop_fit_init


            
    his_best_fit.append(corspod_best_fit)
        
    #fit_copy = fit.copy()
    flag = 0
    for n in range(MaxIter-1):
        print('ASGW Itration : '+str(round)+'-'+str(n)+'  Fitness: '+str(corspod_best_fit)+'  Acc: '+str(corspod_best_acc)+
               '  NumF: '+str(len(corspod_best_cols))+'  Features: '+str(corspod_best_cols))
               
        ind = np.argsort(fit)
        alpha = pop[ind[0]]
        alpha_fit = fit[ind[0]]
        beta = pop[ind[1]]
        beta_fit = pop[ind[1]]
        delta = pop[ind[2]]
        delta_fit = fit[ind[2]]
        
        
        a=2-n*((2)/MaxIter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        
        if(n == 0):
            fit_copy = fit.copy()
            pop = GWO(pop, alpha, beta, delta, pop_size, dimension, a)    
            for i in range(pop_size):
                for j in range(dimension):
                    if (sigmoid(pop[i][j]) > random.random()):
                        pop[i][j] = 1
                    else:
                        pop[i][j] = 0
                        
            for i in range(pop_size):
                new_fit,new_acc,new_cols = Fit_KNN(pop[i], trainX, testX, trainy, testy)
                fit[i]=new_fit
                if (new_fit<corspod_best_fit):
                    corspod_best_fit=new_fit
                    corspod_best_acc=new_acc
                    corspod_best_cols=new_cols
                
            if(fraction(fit,fit_copy) > dimension/2):
                flag = 0
            else:
                flag = 1
                        
            n = n+1
                    
        
            
        
        if(flag == 0):
            fit_copy = fit.copy()
            pop = GWO(pop, alpha, beta, delta, pop_size, dimension, a)
            for i in range(pop_size):
                for j in range(dimension):
                    if (sigmoid(pop[i][j]) > random.random()):
                        pop[i][j] = 1
                    else:
                        pop[i][j] = 0
                        
            for i in range(pop_size):
                fit[i],temp2,temp3 = Fit_KNN(pop[i], trainX, testX, trainy, testy)
                
            if(fraction(fit,fit_copy) > dimension/2):
                flag = 0
            else:
                flag = 1
            
        elif (flag == 1):
            fit_copy = fit.copy()
            pop = WOA(pop, pop_size, alpha, a)
            for i in range(pop_size):
                for j in range(dimension):
                    if (sigmoid(pop[i][j]) > random.random()):
                        pop[i][j] = 1
                    else:
                        pop[i][j] = 0   
                        
            for i in range(pop_size):
                new_fit,new_acc,new_cols = Fit_KNN(pop[i], trainX, testX, trainy, testy)
                fit[i]=new_fit
                if (new_fit<corspod_best_fit):
                    corspod_best_fit=new_fit
                    corspod_best_acc=new_acc
                    corspod_best_cols=new_cols
                
                
            if(fraction(fit,fit_copy) > dimension/2):
                flag = 1
            else:
                flag = 0
        his_best_fit.append(corspod_best_fit)
            
    return   corspod_best_fit,his_best_fit,corspod_best_acc,corspod_best_cols
