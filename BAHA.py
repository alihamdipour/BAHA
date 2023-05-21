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
import timeit
from config import *

def BAHA(dim,max_it, npop,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):
    lb = 0
    ub = 1
    pop_pos=np.zeros((npop, dim),dtype=int)
    for i in range(npop):
        for j in range(dim):
            if pop_pos_init[i][j]>0.5:
                pop_pos[i][j]=1
            else:
                pop_pos[i][j]=0                
        
    
    
    pop_fit = np.copy(pop_fit_init)
    temp_acc = np.zeros(npop)
    temp_cols = np.empty(npop, dtype=object)
    corspod_best_acc = best_acc_init
    corspod_best_cols=np.copy(best_cols_init)
    new_acc=float('inf')
    new_cols=[]
    

    best_f = best_fit_init
    best_x = []
    his_best_fit = []
    for i in range(npop):
        if pop_fit[i] <= best_f:
            best_f = pop_fit[i]
            best_x = pop_pos[i, :]
    his_best_fit.append(best_f)
    
    visit_table = np.zeros((npop, npop))
    diag_ind = np.diag_indices(npop)
    visit_table[diag_ind] = float('nan')
    
    for it in range(max_it-1):
        print('Itration : '+str(it)+'-'+str(roound) +'  Fitness: '+str(best_f)+'  Acc: '+str(corspod_best_acc)+
              '  NumF: '+str(len(corspod_best_cols))+'  Features: '+str(corspod_best_cols))
        # Direction
        visit_table[diag_ind] = float('-inf')
        for i in range(npop):
            direct_vector = np.zeros((npop, dim),dtype=int)
            r = np.random.rand()
            # Diagonal flight
            if r < 1 / 3:
                rand_dim = np.random.permutation(dim)
                if dim >= 3:
                    rand_num = np.ceil(np.random.rand() * (dim - 2))
                else:
                    rand_num = np.ceil(np.random.rand() * (dim - 1))

                direct_vector[i, rand_dim[:int(rand_num)]] = 1
            # Omnidirectional flight
            elif r > 2 / 3:
                direct_vector[i, :] = 1
            else:
                # Axial flight
                rand_num = ceil(np.random.rand() * (dim - 1))
                direct_vector[i, int(rand_num)] = 1
            # Guided foraging
            if np.random.rand() < 0.5:
                MaxUnvisitedTime = max(visit_table[i, :])
                TargetFoodIndex = visit_table[i, :].argmax()
                MUT_Index = np.where(visit_table[i, :] == MaxUnvisitedTime)
                if len(MUT_Index[0]) > 1:
                    Ind = pop_fit[MUT_Index].argmin()
                    TargetFoodIndex = MUT_Index[0][Ind]
                    
                muted=mutation(np.bitwise_xor(pop_pos[i, :],pop_pos[TargetFoodIndex, :]),it,max_it)
                m0=np.bitwise_xor(pop_pos[i, :],muted)
                m1=np.bitwise_and(pop_pos[i, :],muted)            
                z0,z1,z2=Fit_KNN(m0, trainX, testX, trainy, testy)
                x0,x1,x2=Fit_KNN(m1, trainX, testX, trainy, testy)               
                if z0<x0:
                     newPopPos =m0
                else:
                     newPopPos =m1 
                            
                newPopFit,new_acc,new_cols = Fit_KNN(newPopPos,trainX, testX, trainy, testy)
                if newPopFit < pop_fit[i]:
                    pop_fit[i] = newPopFit
                    pop_pos[i, :] = newPopPos
                    temp_acc[i] = new_acc
                    temp_cols[i]=new_cols
                    visit_table[i, :] += 1
                    visit_table[i, TargetFoodIndex] = 0
                    visit_table[:, i] = np.max(visit_table, axis=1) + 1
                    visit_table[i, i] = float('-inf')
                else:
                    visit_table[i, :] += 1
                    visit_table[i, TargetFoodIndex] = 0
            else:
                # Territorial foraging
                muted=mutation(np.bitwise_xor(direct_vector[i, :] , pop_pos[i, :]),it,max_it)
                newPopPos= np.bitwise_xor(pop_pos[i, :],muted)
                m0=np.bitwise_xor(pop_pos[i, :],muted)
                m1=np.bitwise_and(pop_pos[i, :],muted)            
                z0,z1,z2=Fit_KNN(m0, trainX, testX, trainy, testy)
                x0,x1,x2=Fit_KNN(m1, trainX, testX, trainy, testy)               
                if z0<x0:
                     newPopPos =m0
                else:
                     newPopPos =m1 

                
                newPopFit,new_acc,new_cols =Fit_KNN(newPopPos,trainX, testX, trainy, testy)
                if newPopFit < pop_fit[i]:
                    pop_fit[i] = newPopFit
                    pop_pos[i, :] = newPopPos
                    temp_acc[i] = new_acc
                    temp_cols[i]=new_cols
                    visit_table[i, :] += 1
                    visit_table[:, i] = np.max(visit_table, axis=1) + 1
                    visit_table[i, i] = float('-inf')
                else:
                    visit_table[i, :] += 1
        visit_table[diag_ind] = float('nan')
        # Migration foraging
        if np.mod(it, 2 * npop) == 0:
            visit_table[diag_ind] = float('-inf')
            MigrationIndex = pop_fit.argmax()
            pop_pos[MigrationIndex, :] = np.random.binomial(1,0.5,dim)
            visit_table[MigrationIndex, :] += 1
            visit_table[:, MigrationIndex] = np.max(visit_table, axis=1) + 1
            visit_table[MigrationIndex, MigrationIndex] = float('-inf')
            pop_fit[MigrationIndex],temp_acc[MigrationIndex],temp_cols[MigrationIndex]= Fit_KNN(pop_pos[MigrationIndex, :],trainX, testX, trainy, testy)
            visit_table[diag_ind] = float('nan')
        for i in range(npop):
            if pop_fit[i] < best_f:
                best_f = pop_fit[i]
                best_x = pop_pos[i, :]
                corspod_best_acc = temp_acc[i]
                corspod_best_cols=temp_cols[i]
        his_best_fit.append(best_f) 
    return best_x, best_f, his_best_fit,corspod_best_acc,corspod_best_cols

def mutation(x,i,max_iter):
    length=len(x)
    r=(max_iter-i)/max_iter*length/3+1
    
    r=int(math.floor(r))
    values = randint(0, length, r)
    for i in values :        
        x[i]=1-x[i]
    return x            

