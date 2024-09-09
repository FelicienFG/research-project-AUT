#!/usr/bin/python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Randomized Multi-DAG Task Generator
# Xiaotian Dai
# Real-Time Systems Group
# University of York, UK
# -------------------------------------------------------------------------------

import networkx as nx
import pickle
import os

def load_task(task_idx, dag_base_folder = "../data/"):
    # << load DAG task <<
    dag_task_file = dag_base_folder + "Tau_{:d}.gpickle".format(task_idx)

    # task is saved as NetworkX gpickle format
    G = nx.path_graph(4)
    with open(dag_task_file, 'rb') as f:
        G = pickle.load(f)

    # formulate the graph list
    G_dict = {}
    C_dict = {}
    V_array = []
    T = G.graph["T"]

    max_key = 0
    for u, v, weight in G.edges(data='label'):
        if u not in G_dict:
            G_dict[u] = [v]
        else:
            G_dict[u].append(v)

        if v > max_key:
            max_key = v

        if u not in V_array:
            V_array.append(u)
        if v not in V_array:
            V_array.append(v)

        C_dict[u] = weight
    C_dict[max_key] = G.nodes[max_key]['C']
    G_dict[max_key] = []

    # formulate the c list (c[0] is c for v1!!)
    C_array = []
    for key in sorted(C_dict):
        C_array.append(C_dict[key])

    V_array.sort()
    W = sum(C_array)

    # read the ET of the sink node
    # C = G.nodes[max_key]['C']
    # print(C)

    # >> end of load DAG task >>
    return G_dict, V_array, C_dict, C_array, T, W


def outputExceedingNodesDagTasks(maxNodes = 20):
    numberOfTasks = len(os.listdir("data/")) // 3

    with open("tasks_to_remove", "w+") as outputFile:
        for id in range(numberOfTasks):
            _, V, _, _, _, _ = load_task(id, dag_base_folder="data/")
            if len(V) > maxNodes:
                outputFile.write("%i\n" % (id))

def outputLowNodesDagTasks(minNodes = 20):
    numberOfTasks = len(os.listdir("data/")) // 3

    with open("tasks_to_remove", "a") as outputFile:
        for id in range(numberOfTasks):
            _, V, _, _, _, _ = load_task(id, dag_base_folder="data/")
            if len(V) < minNodes:
                outputFile.write("%i\n" % (id))

# below is an example of how to use the load function:
if __name__ == "__main__":
    #G, V, C, _, T, W = load_task(task_idx=0, dag_base_folder="./data/")

    #print("G: ", G)
    #print("V: ", V)
    #print("T: ", T)
    #print("C: ", C)
    #print("W: ", W)
    outputExceedingNodesDagTasks(15)
    outputLowNodesDagTasks(15)