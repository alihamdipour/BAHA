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
import matplotlib.ticker as ticker
from matplotlib.axis import Axis
from matplotlib.ticker import ScalarFormatter
import copy




from config import *
from  AHA import *
from  AIEOU import *
from  WOA_CM import *
from  BAHA import *
from  RTHS import *
from  ASGW import *
from  BSNDO import *


def makeAxisNumberList(down, up, tickNumbers, base=2):
    difference = up - down + 1
    ticks = [up]
    for i in range(1, tickNumbers-1):
        difference /= base
        ticks.append(ticks[len(ticks)-1] - difference)
    ticks.append(down)
    ticks.reverse()
    return ticks


def main():

    folder='out-baha-aha1'
    
    max_it = 3
    npop = 3
    max_avrage=1
    
    datasetList = ["BreastCancer.csv", "BreastEW.csv"]


    randomstateList=[15,5,15,26,12,7,10,8,37,19,2,49,26,1,25,47,12,35] 

    algorthims=['AHA=1','BAHA=1']
    
    for datasetinx in range(len(datasetList)):
        
        avg_best_x_2,avg_best_f_2,avg_his_best_fit_2,avg_best_acc_2,avg_best_cols_2,avg_num_featuers_2,avg_time_2=[],[],[],[],[],[],[]
        avg_best_x_8,avg_best_f_8,avg_his_best_fit_8,avg_best_acc_8,avg_best_cols_8,avg_num_featuers_8,avg_time_8=[],[],[],[],[],[],[]

        for avg in range(max_avrage):
            dataset=datasetList[datasetinx]
            dataset = dataset[:-4]
            print(dataset)
            data,label=read_data(dataset+'.csv')   
            trainX, testX, trainy, testy=data_split(data,label,randomstateList[datasetinx])
            dim=len(trainX[0])
            print('dim= '+str(dim))
            acc0,cols0=orginall_acc(dim,trainX, testX, trainy, testy)


            pop_pos_init = np.zeros((npop, dim))
            pop_fit_init =np.zeros(npop)
            best_fit_init=float('inf')
            best_cols_init=[]
            best_pop_init=[]
            best_acc_init=float('inf')
            for i in range(dim):
                pop_pos_init[:, i] = np.random.rand(npop)            
            for i in range(npop):
                pop_fit_init[i],tempAcc,tempCols= Fit_KNN(pop_pos_init[i, :],trainX, testX, trainy, testy)
                if  best_fit_init>pop_fit_init[i]:
                    best_fit_init=pop_fit_init[i]
                    best_pop_init=pop_pos_init[i, :]
                    best_acc_init=tempAcc
                    best_cols_init=tempCols

            if 'AHA=1'  in algorthims:
                print('-----------------------------AHA--------------------------')
                start2 = timeit.default_timer()
                best_x2, best_f2, his_best_fit2,best_acc2,best_cols2 = AHA(dim, max_it, npop, trainX, testX, trainy, testy,deepcopy(pop_pos_init),deepcopy(pop_fit_init),deepcopy(best_pop_init),deepcopy(best_fit_init),deepcopy(best_acc_init),deepcopy(best_cols_init),avg)
                stop2 = timeit.default_timer()
                avg_best_x_2.append(best_x2)
                avg_best_f_2.append(best_f2)
                avg_his_best_fit_2.append(his_best_fit2)
                avg_best_acc_2.append(best_acc2)
                avg_best_cols_2.append(best_cols2)
                avg_num_featuers_2.append(len(best_cols2))
                avg_time_2.append(stop2-start2)

            if 'BAHA=1'  in algorthims:    
                print('---------------------------------BAHA--------------------------')
                start8 = timeit.default_timer()
                best_x8, best_f8, his_best_fit8,best_acc8,best_cols8 = BAHA(dim, max_it, npop, trainX, testX, trainy, testy,deepcopy(pop_pos_init),deepcopy(pop_fit_init),deepcopy(best_pop_init),deepcopy(best_fit_init),deepcopy(best_acc_init),deepcopy(best_cols_init),avg)
                stop8 = timeit.default_timer()
                avg_best_x_8.append(best_x8)
                avg_best_f_8.append(best_f8)
                avg_his_best_fit_8.append(his_best_fit8)
                avg_best_acc_8.append(best_acc8)
                avg_best_cols_8.append(best_cols8)
                avg_num_featuers_8.append(len(best_cols8))
                avg_time_8.append(stop8-start8)

       
         
        print('************************* The dataset : ', dataset,'*************************')
        print('Orginall features #Num: ', len(cols0))
        print('Orginall Acc: ', acc0)
        f_acc = open(folder+"/acc.txt", "a")
        f_feature= open(folder+"/feature.txt", "a")
        f_time=open(folder+"/time.txt", "a")
        f_fit=open(folder+"/fit.txt", "a")
        f_cave=open(folder+'/caves/'+datasetList[datasetinx], "a")
        
        f_acc.write(dataset+',')
        f_feature.write(dataset+',')
        f_time.write(dataset+',')
        f_fit.write(dataset+',')
        
        if 'BAHA=1'  in algorthims:
            f_acc.write("{:.2f}".format(Average(avg_best_acc_8))+',')
            f_fit.write("{:.2f}".format(Average(avg_best_f_8))+',')
            f_feature.write("{:.2f}".format(Average(avg_num_featuers_8))+',')
            f_time.write("{:.2f}".format(Average(avg_time_8))+',')
            f_cave.write(str(np.average(avg_his_best_fit_8, axis=0))+'\n')
        if 'AHA=1'  in algorthims:
            f_acc.write("{:.2f}".format(Average(avg_best_acc_2))+',')
            f_fit.write("{:.2f}".format(Average(avg_best_f_2))+',')
            f_feature.write("{:.2f}".format(Average(avg_num_featuers_2))+',')
            f_time.write("{:.2f}".format(Average(avg_time_2))+',')
            f_cave.write(str(np.average(avg_his_best_fit_2, axis=0))+'\n')


        f_acc.close()
        f_fit.close()
        f_feature.close()
        f_time.close()
        f_cave.close()

        
        if 'AHA=1'  in algorthims:            
            plot(arange(1,len(his_best_fit2)+1), np.average(avg_his_best_fit_2, axis=0), 'b', label='AHA')            
        if 'BAHA=1'  in algorthims:            
            plot(arange(1,  len(his_best_fit8)+1),np.average(avg_his_best_fit_8, axis=0), 'g', label='BAHA')########### b g r c m y k


        # yscale('log')
        yTicks = makeAxisNumberList()
        plt.xlim([0, max_it + 1])
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.title(dataset)   
        
        handles, labels = plt.gca().get_legend_handles_labels()        
        legend()
        
        
        savefig(folder+'\\'+'images\\'+dataset+'.png')
        clf()

def orginall_acc(dim,trainX, testX, trainy, testy):
    fit,acc,cols=Fit_KNN([1 for i in range(0,dim)],trainX, testX, trainy, testy)
    return acc,cols
def Average(lst):
    return sum(lst) / len(lst)


if __name__ == '__main__':
    main()
