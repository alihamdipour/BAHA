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

def Ufunc(gamma, alpha, beta):
    return alpha * abs(pow(gamma, beta))
def toBinary(currAgent):
    Xnew = np.zeros(np.shape(currAgent))
    for i in range(np.shape(currAgent)[0]):
        random.seed(time.time()+i)
        temp = Ufunc(currAgent[i],2,4)
        if temp > 0.5: # sfunction
            Xnew[i] = float(1)
        else:
            Xnew[i] = float(0)
        
    return Xnew 
 
def adaptiveBeta(agent, agentFit, trainX,trainy,testX,testy,best_acc,best_cols):

    

    new_acc=float('inf')
    new_cols=[]
    
    bmin = 0.1 #parameter: (can be made 0.01)
    bmax = 1
    maxIter = 10 # parameter: (can be increased )
    
    for curr in range(maxIter):
        neighbor = agent.copy()
        size = len(neighbor)
        neighbor = randomwalk(neighbor)

        beta = bmin + (curr / maxIter)*(bmax - bmin)
        for i in range(size):
            random.seed( time.time() + i )
            if random.random() <= beta:
                neighbor[i] = agent[i]
        neighFit,new_acc,new_cols= Fit_KNN(neighbor,trainX,testX,trainy,testy)
        if neighFit <= agentFit:
            agent = neighbor.copy()
            agentFit = neighFit
            best_acc = new_acc
            best_cols=new_cols
        
    return (agent,agentFit,best_acc,best_cols)

def signFunc(x): #signum function? or just sign ?
    if x<0:
        return -1
    return 1

def avg_concentration(eqPool,poolSize,dimension):
    # simple average
    (r,) = np.shape(eqPool[0])
    avg = np.zeros(np.shape(eqPool[0]))
    for i in range(poolSize):
        x = np.array(eqPool[i])
        avg = avg + x

    avg = avg/poolSize

    for i in range(dimension):
        if avg[i]>=0.5:
            avg[i] = 1
        else:
            avg[i] = 0
    return avg
def updateLA(prevDec,beta,pvec):
    a= 0.01
    b= 0.01
    r=3
    if beta==0: 
        for j in range(3): 
            if j-1 == prevDec:
                pvec[j]=pvec[j]+a*(1-pvec[j])
            else:
                pvec[j]=(1-a)*pvec[j]
    elif beta==1: 
        for j in range(3): 
            if j-1 == prevDec:
                pvec[j]=(1-b)*pvec[j]
            else:
                pvec[j]= b/(r-1)+ (1-b)*pvec[j]
    return pvec

def randomwalk(agent):
    percent = 30
    percent /= 100
    neighbor = agent.copy()
    size = len(agent)
    upper = int(percent*size)
    if upper <= 1:
        upper = size
    x = random.randint(1,upper)
    pos = random.sample(range(0,size - 1),x)
    for i in pos:
        neighbor[i] = 1 - neighbor[i]
    return neighbor

def AIEOU(dimension,maxIter, popSize,trainX, testX, trainy, testy, pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,round):#       dataset, randomstate, al, beta

    his_best_fit=[]
    poolSize = 4
    Amax=5
    Amin=0.1
    GPmax = 1
    GPmin = 0
    deltaA2=0.5
    deltaA1=0.5
    deltaGP=0.05
    #can be tuned: t, GP,

    clf=KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainX,trainy)
 
    temp_acc = np.zeros(popSize)
    temp_cols = np.empty(popSize, dtype=object)
  
    best_acc = best_acc_init
    best_cols=best_cols_init
    
    new_acc=float('inf')
    new_cols=[]

    LAA1 = np.zeros((popSize,3))
    LAA2 = np.zeros((popSize,3))
    LAGP = np.zeros((popSize,3))
    A1=np.zeros(popSize)
    A2=np.zeros(popSize)
    GP=np.zeros(popSize)
    for i in range(popSize):
        LAA1[i][0] = (1/3)
        LAA1[i][1] = (1/3)
        LAA1[i][2] = (1/3)

        LAA2[i][0] = (1/3)
        LAA2[i][1] = (1/3)
        LAA2[i][2] = (1/3)

        LAGP[i][0] = (1/3)
        LAGP[i][1] = (1/3)
        LAGP[i][2] = (1/3)


        A1[i]=(Amax+Amin)/2
        A2[i]=(Amax+Amin)/2
        GP[i]=(GPmax+GPmin)/2

    eqPool = np.zeros((poolSize+1,dimension))
    eqfit = np.zeros(poolSize+1)
    for i in range(poolSize+1):
        eqfit[i] = 100

    population=pop_pos_init
    fitList= pop_fit_init    
    his_best_fit.append(best_fit_init)
    best_f=best_fit_init
     
     
    for curriter in range(maxIter-1):

        print('AIEOU, itration : '+str(round)+'-'+str(curriter)+'  Fitness: '+str(best_f)+'  Acc: '+str(best_acc)+
              '  NumF: '+str(len(best_cols))+'  Features: '+str(best_cols))
        
        popnew = np.zeros((popSize,dimension))
        for i in range(popSize):
            for j in range(poolSize):
                if fitList[i] <= eqfit[j]:
                    eqfit[j] = deepcopy(fitList[i])
                    eqPool[j] = population[i].copy()
                    # if j==0:
                    #     best_acc = temp_acc[i]
                    #     best_cols=temp_cols[i]
                    break

        Cave = avg_concentration(eqPool,poolSize,dimension)
        eqPool[poolSize] = Cave.copy()
        eqfit[poolSize],new_acc,new_cols = Fit_KNN(Cave,trainX,testX,trainy,testy) #newPopFit,new_acc,new_cols    
        
        for p in range(len(eqPool)):
            eqPool[p], eqfit[p],new_acc,new_cols = adaptiveBeta(eqPool[p], eqfit[p], trainX,trainy,testX,testy,best_acc,best_cols)
            if p==0:

                his_best_fit.append(eqfit[p])
                best_x=eqPool[p]
                best_f=eqfit[p]
                best_acc=new_acc
                best_cols=new_cols

            
        fitListnew=[]
        for i in range(popSize):
            #choose THE BEST candidate from the equillibrium pool
            bfit = eqfit[0]
            bcan = eqPool[0]
            for e in range(1,len(eqPool)):
                if eqfit[e] < bfit:
                    bfit = eqfit[e]
                    bcan = eqPool[e]
            
            Ceq = bcan
            
            lambdaVec = np.zeros(np.shape(Ceq))
            rVec = np.zeros(np.shape(Ceq))
            for j in range(dimension):
                random.seed(time.time() + 1.1)
                lambdaVec[j] = random.random()
                random.seed(time.time() + 10.01)
                rVec[j] = random.random()

            random.seed(time.time()+17)
            decisionGP = np.random.choice([-1,0,1],1,p=LAGP[i])[0]
            GP[i] = GP[i] + decisionGP*deltaGP
            if GP[i]>GPmax:
                GP[i]=GPmax
            if GP[i]<GPmin:
                GP[i]=GPmin


            random.seed(time.time()+17)
            decisionA1 = np.random.choice([-1,0,1],1,p=LAA1[i])[0]
            A1[i] = A1[i] + decisionA1*deltaA1
            if A1[i]>Amax:
                A1[i]=Amax
            if A1[i]<Amin:
                A1[i]=Amin

            random.seed(time.time()+19)
            decisionA2 = np.random.choice([-1,0,1],1,p=LAA2[i])[0]
            A2[i] = A2[i] + decisionA2*deltaA2
            if A2[i]>Amax:
                A2[i]=Amax
            if A2[i]<Amin:
                A2[i]=Amin

            t = (1 - (curriter/maxIter))**(A2[i]*curriter/maxIter)
            FVec = np.zeros(np.shape(Ceq))
            for j in range(dimension):
                x = -1*lambdaVec[j]*t 
                x = math.exp(x) - 1
                x = A1[i] * signFunc(rVec[j] - 0.5) * x

            random.seed(time.time() + 200)
            r1 = random.random()
            random.seed(time.time() + 20)
            r2 = random.random()
            if r2 < GP[i]:
                GCP = 0
            else:
                GCP = 0.5 * r1
            G0 = np.zeros(np.shape(Ceq))
            G = np.zeros(np.shape(Ceq))
            for j in range(dimension):
                G0[j] = GCP * (Ceq[j] - lambdaVec[j]*population[i][j])
                G[j] = G0[j]*FVec[j]
            temp=[]
            for j in range(dimension):
                temp.append(Ceq[j] + (population[i][j] - Ceq[j])*FVec[j] + G[j]*(1 - FVec[j])/lambdaVec[j])
            temp=np.array(temp)
            popnew[i]=toBinary(temp)
            fitNew,temp_acc,temp_cols = Fit_KNN( popnew[i],trainX,testX,trainy,testy)    #newPopFit,new_acc,new_cols
            if fitNew <best_f:
                best_acc=temp_acc
                best_cols=temp_cols
            fitListnew.append(fitNew)
            beta=1 
            if fitNew<=fitList[i]:
                beta = 0
            LAA1[i]= deepcopy(updateLA(decisionA1,beta,LAA1[i]))
            LAA2[i]= deepcopy(updateLA(decisionA2,beta,LAA2[i]))
            LAGP[i]= deepcopy(updateLA(decisionGP,beta,LAGP[i]))
            

        population = popnew.copy()
        fitList = deepcopy(fitListnew)
        bestfit=[]
        for pop in population:
            newPopFit,new_acc,new_cols=Fit_KNN(pop,trainX,testX,trainy,testy)
            bestfit.append(newPopFit) #newPopFit,new_acc,new_cols

    return  best_x, best_f, his_best_fit,best_acc,best_cols
