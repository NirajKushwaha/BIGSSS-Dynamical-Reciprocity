"""
This script can be run to plot a phase space diagram for a system of agents which evolve using our
propensity updation rule (as explained in the BIGSSS 2022 summer school presentation).

*** Detailed documentation will be provided after the project has been finalized. ***
*** This beta version of the code is only for reference for the participants of the BIGSSS 2022 summer school on Social Cohension. ***
*** Note: This is the first draft of the tested code and so it has not been optimized for performance yet. ***

This code runs parallely on all the cpu threads. If you wish to use some specific number of cpu threads then set the 
cpu_count in the Pool function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

import random

import numpy_indexed

import itertools
from itertools import product
from multiprocess import Pool,cpu_count

def ab_model(number_of_individuals,
             number_of_neighbors,
             number_of_steps,
             propensity_cooperate_arr,
             propensity_accept_arr,
             propensity_change):
    
    memory_list = [[] for i in range(number_of_steps)]
    ### memory_list[step_num][agent_num][neighbor_num][whose decision?] ###
    neighbors_list = [[] for i in range(number_of_steps)]
    ### neighbors_list[step_num][agent_num][neighbor_num]  ###
    accept_memory_list = []
    
    propensity_accept_list = []
    propensity_accept_list.append(np.copy(propensity_accept_arr))
    propensity_cooperate_list = []
    propensity_cooperate_list.append(np.copy(propensity_cooperate_arr))
    
    for step in range(number_of_steps):
        if(step == 0):
            for agent in range(number_of_individuals):
                memory_list[step].append([])
                neighbors_list[step].append([])
                
                neighbors_temp = np.random.choice(np.delete(np.arange(number_of_individuals) , agent),#Randomly selecting neighbors to interact
                                                  size=number_of_neighbors,
                                                  replace=False)
                
                memory_list_temp = []
                neighbors_list_temp = []
                for neighbor in neighbors_temp:
                    decision_agent = np.random.choice(np.arange(0,2),
                                                        p=[1-propensity_cooperate_arr[agent],
                                                           propensity_cooperate_arr[agent]])
                    
                    if(decision_agent == 1):
                        decision_neighbor = np.random.choice(np.arange(0,2),
                                                                p=[1-propensity_accept_arr[neighbor],
                                                                   propensity_accept_arr[neighbor]])
                    else:
                        decision_neighbor = 0
                    
                    memory_list_temp.append((decision_agent,decision_neighbor))
                    
                    neighbors_list_temp.append(neighbor)
                    
                memory_list[step][agent] = memory_list_temp
                neighbors_list[step][agent] = neighbors_list_temp
                
                
        else:
            ### Creating a list of list of offered agents ###
            accept_memory = [[] for i in range(number_of_individuals)]
            for agent_num in range(number_of_individuals):
                for neighbor_num in range(number_of_neighbors):
                    accept_memory[neighbors_list[step-1][agent_num][neighbor_num]].\
                                                append(memory_list[step-1][agent_num][neighbor_num])
                    
            accept_memory_list.append(accept_memory)
            ### accept memory shows tuples (a,b) such that a is cooperation receievd by an agent and b is the response(accept or reject)
            ### that the agent gave to that cooperation.
            
            ### Updating propensities according to interactions ###
            for agent in range(number_of_individuals):
                neighbors_who_accepted = list(zip(*memory_list[step-1][agent]))[1].count(1)
                neighbors_who_rejected = list(zip(*memory_list[step-1][agent]))[1].count(0)
                didnt_offer_cooperation = list(zip(*memory_list[step-1][agent]))[0].count(0)
                
                temp = neighbors_who_accepted-(neighbors_who_rejected-didnt_offer_cooperation)
                
                if(((propensity_accept_arr[agent] + temp*propensity_change) <= 1) & \
                                  ((propensity_accept_arr[agent] + temp*propensity_change) >= 0)):
                    
                    propensity_accept_arr[agent] += temp*propensity_change
                else:
                    if(temp >= 1):
                        propensity_accept_arr[agent] = 1
                    else:
                        propensity_accept_arr[agent] = 0
                
                
                
                if(len(accept_memory_list[step-1][agent]) != 0):
                    got_cooperation_from = list(zip(*accept_memory_list[step-1][agent]))[0].count(1)
                    got_noncooperation_from = list(zip(*accept_memory_list[step-1][agent]))[0].count(0)
                    
                    temp = got_cooperation_from-got_noncooperation_from
                    
                    if(((propensity_cooperate_arr[agent] + temp*propensity_change) <= 1) & \
                                          ((propensity_cooperate_arr[agent] + temp*propensity_change) >= 0)):
                        
                        propensity_cooperate_arr[agent] += temp*propensity_change
                    else:
                        if(temp >= 1):
                            propensity_cooperate_arr[agent] = 1
                        else:
                            propensity_cooperate_arr[agent] = 0
    
                
            propensity_accept_list.append(np.copy(propensity_accept_arr))
            propensity_cooperate_list.append(np.copy(propensity_cooperate_arr))
            
            ### Next round of interactions ###
            for agent in range(number_of_individuals):
                memory_list[step].append([])
                neighbors_list[step].append([])
                
                neighbors_temp = np.random.choice(np.delete(np.arange(number_of_individuals) , agent),#Randomly selecting neighbors to interact
                                                  size=number_of_neighbors,
                                                  replace=False)
                
                memory_list_temp = []
                neighbors_list_temp = []
                for neighbor in neighbors_temp:
                    decision_agent = np.random.choice(np.arange(0,2),
                                                        p=[1-propensity_cooperate_arr[agent],
                                                           propensity_cooperate_arr[agent]])
                    
                    if(decision_agent == 1):
                        decision_neighbor = np.random.choice(np.arange(0,2),
                                                                p=[1-propensity_accept_arr[neighbor],
                                                                   propensity_accept_arr[neighbor]])
                    else:
                        decision_neighbor = 0
                    
                    memory_list_temp.append((decision_agent,decision_neighbor))
                    
                    neighbors_list_temp.append(neighbor)
                    
                memory_list[step][agent] = memory_list_temp
                neighbors_list[step][agent] = neighbors_list_temp
                
    return memory_list,accept_memory_list,neighbors_list,propensity_cooperate_list,propensity_accept_list


#### Parallel cpu threading ####
def system_state_loop_wrapper(args):
    propensity_tuple = args
    
    propensity_cooperate_arr = np.array([propensity_tuple[0] for i in range(number_of_individuals)])
    propensity_accept_arr = np.array([propensity_tuple[1] for i in range(number_of_individuals)])
    
    memory_list,\
    accept_memory_list,\
    neighbors_list,\
    propensity_cooperate_list,\
    propensity_accept_list = ab_model(number_of_individuals,
                                         number_of_neighbors,
                                            number_of_steps,
                                             propensity_cooperate_arr,
                                                propensity_accept_arr,
                                                    propensity_change)  
    
    propensity_pairs = np.column_stack((propensity_cooperate_list[-1],propensity_accept_list[-1])).astype(int)

    type_list = [0,0,0,0]   #00,01,10,11
    for propensity_pair in propensity_pairs:
        if(np.array_equal(propensity_pair , np.array([0,0]))):
            type_list[0] += 1
        elif(np.array_equal(propensity_pair , np.array([0,1]))):
            type_list[1] += 1
        elif(np.array_equal(propensity_pair , np.array([1,0]))):
            type_list[2] += 1
        elif(np.array_equal(propensity_pair , np.array([1,1]))):
            type_list[3] += 1
            
    system_state = 0
    for i,weight in enumerate([-2,-1,1,2]):
        system_state += type_list[i]*weight
        
    return system_state

### Function definitions ended ###

## Set parameters here ##
number_of_individuals = 50
number_of_steps = 1000
number_of_neighbors = 4
propensity_change = 0.001

resolution = 0.04
parameter_ticks = np.arange(0,1+resolution,resolution)
##

parameter_pairs = list(product(parameter_ticks,parameter_ticks))
with Pool() as pool:
    output = list(pool.map(system_state_loop_wrapper , parameter_pairs))

system_state_matrix = np.zeros((len(parameter_ticks) , len(parameter_ticks))).astype(int)

k = 0
for i in range(len(parameter_ticks)):
    for j in range(len(parameter_ticks)):
        system_state_matrix[j][i] = output[k]
        k += 1

plt.imshow(system_state_matrix , origin="lower" , interpolation="bicubic")
#a = plt.imshow(system_state_matrix , origin="lower")

cbar = plt.colorbar(ticks=[-100,-50,50,100])
cbar.ax.set_yticklabels(["(0.0,0.0)" , "(0.0,1.0)" , "(1.0,0.0)" , "(1.0,1.0)"] , fontsize=13)

plt.xticks(ticks=[0,5,10,15,20,25], labels=[0,0.2,0.4,0.6,0.8,1]) #For resolution = 0.04 
plt.yticks(ticks=[0,5,10,15,20,25], labels=[0,0.2,0.4,0.6,0.8,1]) #For resolution = 0.04 

plt.xlabel("Propensity_help" , fontsize=14)
plt.ylabel("Propensity_accept" , fontsize=14)
plt.title(f"Propensity_change={str(propensity_change)} , steps={str(number_of_steps)}" , fontsize=13)

plt.show()
