# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:46:56 2022

@author: Driver code for using the BanditProblem class
This file demonstrates how to solve the multi-armed bandit problem by using 
the epsilon-greedy action value approach

"""
# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from BanditProblem import BanditProblem

# these are the means of the action values that are used to simulate the multi-armed bandit problem
actionValues=np.array([1,4,2,0,7,1,-1])

# epsilon values to investigate the performance of the method
epsilon1=0
epsilon2=0.1
epsilon3=0.2
epsilon4=0.3

# total number of simulation steps 
totalSteps=100000

# create four different bandit problems and simulate the method performance
Bandit1=BanditProblem(actionValues, epsilon1, totalSteps)
Bandit1.playGame()
epsilon1MeanReward=Bandit1.meanReward
Bandit2=BanditProblem(actionValues, epsilon2, totalSteps)
Bandit2.playGame()
epsilon2MeanReward=Bandit2.meanReward
Bandit3=BanditProblem(actionValues, epsilon3, totalSteps)
Bandit3.playGame()
epsilon3MeanReward=Bandit3.meanReward
Bandit4=BanditProblem(actionValues, epsilon4, totalSteps)
Bandit4.playGame()
epsilon4MeanReward=Bandit4.meanReward

#plot the results
plt.plot(np.arange(totalSteps+1),epsilon1MeanReward,linewidth=2, color='r', label='epsilon =0')
plt.plot(np.arange(totalSteps+1),epsilon2MeanReward,linewidth=2, color='k', label='epsilon =0.1')
plt.plot(np.arange(totalSteps+1),epsilon3MeanReward,linewidth=2, color='m', label='epsilon =0.2')
plt.plot(np.arange(totalSteps+1),epsilon4MeanReward,linewidth=2, color='b', label='epsilon =0.3')
plt.xscale("log")
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()
plt.savefig('results.png',dpi=300)
plt.show()


