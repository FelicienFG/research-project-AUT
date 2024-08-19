#!/usr/env python3
import networkx as nx
import pickle
import os
import makespan_solver as ms
import torch
import torch.utils.data.sampler as torch_samplers
import numpy


def load_task(dataFolder, id):
    dag_task_file = dataFolder + "Tau_{:d}.gpickle".format(id)

    # task is saved as NetworkX gpickle format
    G = nx.path_graph(4)
    with open(dag_task_file, 'rb') as f:
        G = pickle.load(f)

    # formulate the graph list
    G_dict = {}
    C_dict = {}
    #V_array = []
    T = G.graph["T"]

    max_key = 0
    for u, v, weight in G.edges(data='label'):
        if u not in G_dict:
            G_dict[u] = {"in": [], "out": [v]}
        else:
            G_dict[u]["out"].append(v)
        if v not in G_dict:
            G_dict[v] = {"in": [u], "out": []}
        else:
            G_dict[v]["in"].append(u)

        if v > max_key:
            max_key = v

        #if u not in V_array:
        #    V_array.append(u)
        #if v not in V_array:
        #    V_array.append(v)

        C_dict[u] = weight

    C_dict[max_key] = G.nodes[max_key]['C']

    # formulate the c list (c[0] is c for v1!!)
    C_array = []
    for key in sorted(C_dict):
        C_array.append(C_dict[key])

    #V_array.sort()
    W = sum(C_array)

    return G_dict, C_dict, T, W

def getDagTask(graph, wcets):
    """
    get the list of DagSubtasks from the graph
    """
    dagTask = ms.DagSubtaskVector()
    for vertex_key in wcets:
        dagSubtask = ms.DagSubtask()
        dagSubtask.id = vertex_key - 1 
        dagSubtask.inDependencies = ms.IntList(graph[vertex_key]['in'])
        for i in range(dagSubtask.inDependencies.size()):
            dagSubtask.inDependencies[i] -= 1
        dagSubtask.outDependencies = ms.IntList(graph[vertex_key]['out'])
        for i in range(dagSubtask.outDependencies.size()):
            dagSubtask.outDependencies[i] -= 1
        dagSubtask.wcet = wcets[vertex_key]

        dagTask.append(dagSubtask)

    return dagTask

def criticalPath(graph, start, wcets):
    if graph[start]['out'] == []: #it's the sink node
        path = [start]
        return path, wcets[start]
    
    #otherwise, search the biggest neighbour's path
    biggest_path = []
    max_length = 0
    for neighbour in graph[start]['out']:
        neigh_biggest_path, length = criticalPath(graph, neighbour, wcets)
        neigh_biggest_path.append(start)
        length += wcets[start]
        if length > max_length:
            biggest_path = neigh_biggest_path
            max_length = length

    return biggest_path, max_length
       

class DataLoader:

    def __init__(self, data_folder, maxNodesPerDag = 30):
        self.numberOfTasks = len(os.listdir(data_folder)) // 3
        self.dataFolder = data_folder
        self.maxNodesPerDag = maxNodesPerDag
        self.tasks = []
        #node features : C_i / W , deg_in, deg_out, is_source_or_sink, is_in_critical_path
        self.taskFeatures = []
        self.dagTasks = []
        self.ilpOutputs = []

        for id in range(self.numberOfTasks):
            G_adjaList, C_dict , T, W = load_task(self.dataFolder, id)
            self.tasks.append({"G": G_adjaList, "C": C_dict, "T": T, "W": W})
            self.dagTasks.append(getDagTask(G_adjaList, C_dict))
            self.addTaskFeatureMatrix(id, G_adjaList, C_dict, W)
        self.taskFeatures = torch.Tensor(self.taskFeatures)
        self.ilpOutputs = torch.Tensor(self.ilpOutputs)

    def addTaskFeatureMatrix(self, id, adjaList, wcets, totalW):
        self.taskFeatures.append([])
        crit_path, crit_length = criticalPath(adjaList, 1, wcets)
        self.ilpOutputs.append(crit_length)
        for node in adjaList:
            is_source_sink = int(len(adjaList[node]['in']) == 0 or len(adjaList[node]['out']) == 0)
            is_in_critical_path = int(node in crit_path)
            self.taskFeatures[id].append([wcets[node] / totalW, len(adjaList[node]['in']), len(adjaList[node]['out']), is_source_sink, is_in_critical_path])
        self.fillZerosTaskFeatureMatrix(id)
        
    def fillZerosTaskFeatureMatrix(self, taskID):
        if len(self.taskFeatures[taskID]) > self.maxNodesPerDag:
            raise RuntimeError("DataLoader: DAG task has more nodes than what is permitted (%i > max = %i)" % (len(self.taskFeatures[taskID]), self.maxNodesPerDag))
        if len(self.taskFeatures[taskID]) < self.maxNodesPerDag:
            nodeDiff = self.maxNodesPerDag - len(self.taskFeatures[taskID])
            for i in range(nodeDiff):
                self.taskFeatures[taskID].append([0,0,0,0,0])
            

    def train_val_split(self, train_percentage = 0.8, batch_size = 2):
        train_threshold = int(train_percentage * self.numberOfTasks)
        batches_indices = list(torch_samplers.BatchSampler(torch_samplers.RandomSampler(self.taskFeatures[:train_threshold]), 
                                                           batch_size=batch_size,drop_last=True))
        train_batches = batches_indices
        val_set = (self.dagTasks[train_threshold:], self.taskFeatures[train_threshold:], self.ilpOutputs[train_threshold:])

        return train_batches, val_set
    
if __name__ == "__main__":

    data = DataLoader("../dag_generator/data/", 24)
    print(data.taskFeatures.shape)
    print(data.taskFeatures[6])
    #print(criticalPath(data.tasks[1]['G'], 1, data.tasks[1]['C']))