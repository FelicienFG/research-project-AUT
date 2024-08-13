#!/usr/env python3
import networkx as nx
import pickle
import os
import makespan_solver as ms
import torch



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

    def __init__(self, data_folder):
        self.numberOfTasks = len(os.listdir(data_folder)) // 3
        self.dataFolder = data_folder
        self.tasks = []
        #node features : C_i / W , deg_in, deg_out, is_source_or_sink, is_in_critical_path
        self.taskFeatures = []
        self.dagTasks = []
        self.ilpOutputs = []

        for id in range(self.numberOfTasks):
            G_adjaList, C_dict , T, W = load_task(self.dataFolder, id)
            self.tasks.append({"G": G_adjaList, "C": C_dict, "T": T, "W": W})
            self.dagTasks.append(getDagTask(G_adjaList, C_dict))
            #add task feature matrix
            self.taskFeatures.append([])
            crit_path, crit_length = criticalPath(G_adjaList, 1, C_dict)
            self.ilpOutputs.append(crit_length)
            for node in G_adjaList:
                is_source_sink = 0
                is_in_critical_path = 0
                if len(G_adjaList[node]['in']) == 0 or len(G_adjaList[node]['out']) == 0:
                    is_source_sink = 1
                if node in crit_path:
                    is_in_critical_path = 1
                self.taskFeatures[id].append([C_dict[node] / W, len(G_adjaList[node]['in']), len(G_adjaList[node]['out']), is_source_sink, is_in_critical_path])

            self.taskFeatures[id] = torch.Tensor(self.taskFeatures[id])

    def train_val_split(self, train_percentage = 0.8):
        train_threshold = int(0.8 * self.numberOfTasks)
        train_set = (self.dagTasks[:train_threshold], self.taskFeatures[:train_threshold], self.ilpOutputs[:train_threshold])
        val_set = (self.dagTasks[train_threshold:], self.taskFeatures[train_threshold:], self.ilpOutputs[train_threshold:])

        return train_set, val_set
    
if __name__ == "__main__":

    data = DataLoader("../dag_generator/data/")
    print(data.taskFeatures)
    #print(criticalPath(data.tasks[1]['G'], 1, data.tasks[1]['C']))