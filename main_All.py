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
import os
import copy
from matplotlib.ticker import StrMethodFormatter, NullFormatter




from config import *
from  AHA import *
from  AIEOU import *
from  WOA_CM import *
from  BAHA import *
from  RTHS import *
from  ASGW import *
from  BSNDO import *


def main():
    os.system("cls")

    max_it =5
    npop = 3
    max_avrage=1
    folder='out1'
    
    datasetList = ["PenglungEW.csv","SpectEW.csv","KrVsKpEW.csv", "WaveformEW.csv"]


    randomstateList=[15,5,15,26,12,7,10,8,37,19,2,49,26,1,25,47,12,35] 

    algorthims=['AIEOU=1','WOA_CM=1','ASGW=1','BAHA=1','RTHS=1','BSNDO=0']


  
     
    
    for datasetinx in range(len(datasetList)):
        
        avg_best_x_2,avg_best_f_2,avg_his_best_fit_2,avg_best_acc_2,avg_best_cols_2,avg_num_featuers_2,avg_time_2=[],[],[],[],[],[],[]
        avg_best_x_3,avg_best_f_3,avg_his_best_fit_3,avg_best_acc_3,avg_best_cols_3,avg_num_featuers_3,avg_time_3=[],[],[],[],[],[],[]
        avg_best_x_5,avg_best_f_5,avg_his_best_fit_5,avg_best_acc_5,avg_best_cols_5,avg_num_featuers_5,avg_time_5=[],[],[],[],[],[],[]
        avg_best_x_6,avg_best_f_6,avg_his_best_fit_6,avg_best_acc_6,avg_best_cols_6,avg_num_featuers_6,avg_time_6=[],[],[],[],[],[],[]
        avg_best_x_7,avg_best_f_7,avg_his_best_fit_7,avg_best_acc_7,avg_best_cols_7,avg_num_featuers_7,avg_time_7=[],[],[],[],[],[],[]
        avg_best_x_8,avg_best_f_8,avg_his_best_fit_8,avg_best_acc_8,avg_best_cols_8,avg_num_featuers_8,avg_time_8=[],[],[],[],[],[],[]
        avg_best_x_9,avg_best_f_9,avg_his_best_fit_9,avg_best_acc_9,avg_best_cols_9,avg_num_featuers_9,avg_time_9=[],[],[],[],[],[],[]

        
        for avg in range(max_avrage):
            dataset=datasetList[datasetinx]
            dataset = dataset[:-4]
            print(dataset)
            data,label=read_data(dataset+'.csv')   
            trainX, testX, trainy, testy=data_split(data,label,randomstateList[datasetinx])
            dim=len(trainX[0])
            print('dim= '+str(dim))
            
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

           
            acc0,cols0=orginall_acc(dim,trainX, testX, trainy, testy)

            if 'BSNDO=1'  in algorthims:
                print('-----------------------------BSNDO--------------------------')
                start2 = timeit.default_timer()
                best_x2, best_f2, his_best_fit2,best_acc2,best_cols2 = BSNDO(dim, max_it, npop, trainX, testX, trainy, testy)
                stop2 = timeit.default_timer()
                avg_best_x_2.append(best_x2)
                avg_best_f_2.append(best_f2)
                avg_his_best_fit_2.append(his_best_fit2)
                avg_best_acc_2.append(best_acc2)
                avg_best_cols_2.append(best_cols2)
                avg_num_featuers_2.append(len(best_cols2))
                avg_time_2.append(stop2-start2)
            if 'AIEOU=1'  in algorthims:
                print('-------------------------------AIEOU------------------------')
                start3 = timeit.default_timer()
                best_x3, best_f3, his_best_fit3,best_acc3,best_cols3 = AIEOU(dim, max_it, npop, trainX, testX, trainy, testy, deepcopy(pop_pos_init),deepcopy(pop_fit_init),deepcopy(best_pop_init),deepcopy(best_fit_init),deepcopy(best_acc_init),deepcopy(best_cols_init),avg)

                stop3 = timeit.default_timer()
                avg_best_x_3.append(best_x3)
                avg_best_f_3.append(best_f3)
                avg_his_best_fit_3.append(his_best_fit3)
                avg_best_acc_3.append(best_acc3)
                avg_best_cols_3.append(best_cols3)
                avg_num_featuers_3.append(len(best_cols3))
                avg_time_3.append(stop3-start3)

            if 'WOA_CM=1'  in algorthims:    
                print('---------------------------------WOA_CM--------------------------')
                start5 = timeit.default_timer()
                best_x5, best_f5, his_best_fit5,best_acc5,best_cols5 = WOA_CM(dim, max_it, npop, trainX, testX, trainy, testy,  deepcopy(pop_pos_init),deepcopy(pop_fit_init),deepcopy(best_pop_init),deepcopy(best_fit_init),deepcopy(best_acc_init),deepcopy(best_cols_init),avg)
                print("#######################################")
                print(pop_fit_init)
                stop5 = timeit.default_timer()
                avg_best_x_5.append(best_x5)
                avg_best_f_5.append(best_f5)
                avg_his_best_fit_5.append(his_best_fit5)
                avg_best_acc_5.append(best_acc5)
                avg_best_cols_5.append(best_cols5)
                avg_num_featuers_5.append(len(best_cols5))
                avg_time_5.append(stop5-start5)

            if 'ASGW=1'  in algorthims:    
                print('---------------------------------ASGW--------------------------')
                start7 = timeit.default_timer()
                best_f7, his_best_fit7,best_acc7,best_cols7 = ASGW(dim, max_it, npop, trainX, testX, trainy, testy,  deepcopy(pop_pos_init),deepcopy(pop_fit_init),deepcopy(best_pop_init),deepcopy(best_fit_init),deepcopy(best_acc_init),deepcopy(best_cols_init),avg)
                print("#######################################")
                print(pop_fit_init)
                stop7= timeit.default_timer()                
                avg_best_f_7.append(best_f7)
                avg_his_best_fit_7.append(his_best_fit7)
                avg_best_acc_7.append(best_acc7)
                avg_best_cols_7.append(best_cols7)
                avg_num_featuers_7.append(len(best_cols7))
                avg_time_7.append(stop7-start7)
            if 'BAHA=1'  in algorthims:    
                print('---------------------------------BAHA--------------------------')
                start8 = timeit.default_timer()
                best_x8, best_f8, his_best_fit8,best_acc8,best_cols8 = BAHA(dim, max_it, npop, trainX, testX, trainy, testy, deepcopy(pop_pos_init),deepcopy(pop_fit_init),deepcopy(best_pop_init),deepcopy(best_fit_init),deepcopy(best_acc_init),deepcopy(best_cols_init) ,avg)
                stop8 = timeit.default_timer()
                avg_best_x_8.append(best_x8)
                avg_best_f_8.append(best_f8)
                avg_his_best_fit_8.append(his_best_fit8)
                avg_best_acc_8.append(best_acc8)
                avg_best_cols_8.append(best_cols8)
                avg_num_featuers_8.append(len(best_cols8))
                avg_time_8.append(stop8-start8)
            if 'RTHS=1'  in algorthims:    
                print('---------------------------------RTHS--------------------------')
                start9 = timeit.default_timer()
                best_x9, best_f9, his_best_fit9,best_acc9,best_cols9 = RTHS(dim, max_it, npop, trainX, testX, trainy, testy,deepcopy(pop_pos_init),deepcopy(pop_fit_init),deepcopy(best_pop_init),deepcopy(best_fit_init),deepcopy(best_acc_init),deepcopy(best_cols_init),avg)             
                stop9= timeit.default_timer()
                avg_best_x_9.append(best_x9)
                avg_best_f_9.append(best_f9)
                avg_his_best_fit_9.append(his_best_fit9)
                avg_best_acc_9.append(best_acc9)
                avg_best_cols_9.append(best_cols9)
                avg_num_featuers_9.append(len(best_cols9))
                avg_time_9.append(stop9-start9)
       
         
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
        if 'BSNDO=1'  in algorthims:
            f_acc.write("{:.2f}".format(Average(avg_best_acc_2))+',')
            f_fit.write("{:.2f}".format(Average(avg_best_f_2))+',')
            f_feature.write("{:.2f}".format(Average(avg_num_featuers_2))+',')
            f_time.write("{:.2f}".format(Average(avg_time_2))+',')
            f_cave.write(str(np.average(avg_his_best_fit_2, axis=0))+'\n')
        if 'AIEOU=1'  in algorthims:
            f_acc.write("{:.2f}".format(Average(avg_best_acc_3))+',')
            f_fit.write("{:.2f}".format(Average(avg_best_f_3))+',')
            f_feature.write("{:.2f}".format(Average(avg_num_featuers_3))+',')
            f_time.write("{:.2f}".format(Average(avg_time_3))+',')
            f_cave.write(str(np.average(avg_his_best_fit_3, axis=0))+'\n')
            
        if 'RTHS=1'  in algorthims:
            f_acc.write("{:.2f}".format(Average(avg_best_acc_9))+',')
            f_fit.write("{:.2f}".format(Average(avg_best_f_9))+',')
            f_feature.write("{:.2f}".format(Average(avg_num_featuers_9))+'\n')
            f_time.write("{:.2f}".format(Average(avg_time_9))+',')
            f_cave.write(str(np.average(avg_his_best_fit_9, axis=0))+'\n')

        if 'WOA_CM=1'  in algorthims:
            f_acc.write("{:.2f}".format(Average(avg_best_acc_5))+',')
            f_fit.write("{:.2f}".format(Average(avg_best_f_5))+',')
            f_feature.write("{:.2f}".format(Average(avg_num_featuers_5))+',')
            f_time.write("{:.2f}".format(Average(avg_time_5))+',')
            f_cave.write(str(np.average(avg_his_best_fit_5, axis=0))+'\n')
        if 'ASGW=1'  in algorthims:
            f_acc.write("{:.2f}".format(Average(avg_best_acc_7))+'\n')
            f_fit.write("{:.2f}".format(Average(avg_best_f_7))+'\n')
            f_feature.write("{:.2f}".format(Average(avg_num_featuers_7))+'\n')
            f_time.write("{:.2f}".format(Average(avg_time_7))+'\n')
            f_cave.write(str(np.average(avg_his_best_fit_7, axis=0))+'\n')

        f_acc.close()
        f_fit.close()
        f_feature.close()
        f_time.close()
        f_cave.close()

        
        if 'BSNDO=1'  in algorthims:            
            plot(arange(1,len(his_best_fit2)+1), np.average(avg_his_best_fit_2, axis=0), 'c', label='BSNDO')            
        if 'AIEOU=1'  in algorthims:            
            plot(arange(1, max_it + 1), np.average(avg_his_best_fit_3, axis=0), 'y', label='AIEOU')
        if 'RTHS=1'  in algorthims:            
            plot(arange(1,  len(his_best_fit9)+1),np.average(avg_his_best_fit_9, axis=0), 'k', label='RTHS')
        if 'WOA_CM=1'  in algorthims:            
            plot(arange(1, max_it + 1), np.average(avg_his_best_fit_5, axis=0), 'm', label='WOA_CM')
        if 'ASGW=1'  in algorthims:            
            plot(arange(1,  len(his_best_fit7)+1),  np.average(avg_his_best_fit_7, axis=0), 'r', label='ASGW')
        if 'BAHA=1'  in algorthims:            
            plot(arange(1,  len(his_best_fit8)+1),np.average(avg_his_best_fit_8, axis=0), 'g', label='BAHA')########### b g r c m y k


        yscale('log')        
        plt.xlim([0, max_it + 1])
        plt.xlabel('Iterations')
        plt.ylabel('Fitness',rotation=90)
        plt.title(dataset)       
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [4,0,1,2,3]
        legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        savefig(folder+'\\'+'images\\'+dataset+'.png')
        clf()

def orginall_acc(dim,trainX, testX, trainy, testy):
    fit,acc,cols=Fit_KNN([1 for i in range(0,dim)],trainX, testX, trainy, testy)
    return acc,cols
def Average(lst):
    return sum(lst) / len(lst)


if __name__ == '__main__':
    main()
