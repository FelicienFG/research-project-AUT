#!/usr/env python3
import networkx as nx
import pickle
import os
import makespan_solver as ms
import torch
import torch.utils.data.sampler as torch_samplers
import numpy
import copy
import json
import parse
import heapq
import random as rand
import sys
import time

def getOptimalPriorityListFromILPscheduleFile(scheduleFileName):
    priorityList = []
    with open(scheduleFileName, "r") as scheduleFile:
        scheduleJSON = json.load(scheduleFile)
        scheduledTasks = scheduleJSON['TaskInstancesStore']
        taskList = []

        for taskJSON in scheduledTasks:
            subtaskID = parse.parse("task_{}", taskJSON['name'])
            subtaskID = int(subtaskID[0])
            subTaskStartTime = taskJSON['value'][0]['executionIntervals'][0]['startTime']
            heapq.heappush(taskList, (subTaskStartTime, subtaskID))
        
        priority = 0
        priorityList = [0 for i in range(len(scheduledTasks))]
        while taskList:
            (_, subtaskID) = heapq.heappop(taskList)
            priorityList[subtaskID] = priority
            priority += 1

    return priorityList

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
        if (u - 1) not in G_dict:
            G_dict[u-1] = {"in": [], "out": [v-1]}
        else:
            G_dict[u-1]["out"].append(v-1)
        if (v-1) not in G_dict:
            G_dict[v-1] = {"in": [u-1], "out": []}
        else:
            G_dict[v-1]["in"].append(u-1)

        if v > max_key:
            max_key = v

        #if u not in V_array:
        #    V_array.append(u)
        #if v not in V_array:
        #    V_array.append(v)

        C_dict[u-1] = weight

    C_dict[max_key-1] = G.nodes[max_key]['C']

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
        dagSubtask.id = vertex_key
        dagSubtask.inDependencies = ms.IntList(graph[vertex_key]['in'])
        dagSubtask.outDependencies = ms.IntList(graph[vertex_key]['out'])
        dagSubtask.wcet = wcets[vertex_key]

        dagTask.append(dagSubtask)

    return dagTask

def test_makespan_compute():
    rand.seed = time.time()
    makespanSolver = ms.MakespanSolver(numberOfCores=2)
    dataLoader = DataLoader("../dag_generator/data/", "../LET-LP-Scheduler/dag_tasks_output_schedules/")
    task = 4
    dagTask = getDagTask(dataLoader.tasks[task]['G'], dataLoader.tasks[task]['C'])
    
    priorities = ms.IntVector([rand.randint(0, dagTask.size()) for i in range(dagTask.size())])
    
    makespan = makespanSolver.computeMakespan(priorities, dagTask)

    print(makespan)

def outputILPSystemJSON(graph, start, wcets, totalW, numCores, dagTaskID, dag_file):
    outputJSON = {}
    outputFile = '../LET-LP-Scheduler/%s/multicore_system_DAGtask_%i.json' % (dag_file, dagTaskID)

    #first add the cores
    outputJSON['CoreStore'] = []
    for core in range(numCores):
        outputJSON['CoreStore'].append({"name": ("c%i" % (core)), "speedup": 1})

    #then add the tasks and edges
    outputJSON['TaskStore'] = []
    outputJSON['DependencyStore'] = []
    for node in graph:
        is_sink = False
        if len(graph[node]['out']) == 0:
            is_sink = True
        outputJSON['TaskStore'].append(
            {
                "name": ("task_%i" % (node)),
                "initialOffset":0,
                "activationOffset":0,
                "duration":wcets[node],
                "period":totalW * totalW,
                "is_sink": is_sink,
                "inputs":[
                    "in"
                ],
                "outputs":[
                    "out"
                ],
                "wcet":wcets[node],
                "acet":wcets[node],
                "bcet":wcets[node],
                "distribution":"Uniform"
            }
        )

        if len(graph[node]['out']) > 0:
            for outNeighbour in graph[node]['out']:
                outputJSON['DependencyStore'].append(
                {
                    "name":("v%iv%i" % (node, outNeighbour)),
                    "source":{
                        "task":("task_%i" % (node)),
                        "port":"out"
                    },
                    "destination":{
                        "task":("task_%i" % (outNeighbour)),
                        "port":"in"
                    }
                })
    
    with open(outputFile, "w") as jsonFile:
        jsonFile.write(json.dumps(outputJSON, indent=4))

    

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
       
def filledUpAdjaListAndWcets(adjalist, wcets, maxNodes):
    filledUpAdjalist = adjalist
    filledUpWcets = wcets
    for n in range(len(wcets), maxNodes):
        filledUpAdjalist[n] = {"in": [], "out": []}
        filledUpWcets[n] = 0

    return filledUpAdjalist, filledUpWcets

def outputAllILPSystemJSON(inputDataFolder, numCores, dag_file):
    numberOfTasks = len(os.listdir(inputDataFolder)) // 2
    #if the output folder doesn't exist, then create it first
    src_path = os.path.abspath(os.path.dirname(__file__))
    scheduler_path = os.path.abspath(os.path.join(src_path, os.pardir, 'LET-LP-Scheduler'))


    dag_file_path = os.path.join(scheduler_path, dag_file)
    if not os.path.exists(dag_file_path):
        os.makedirs(dag_file_path)

    for id in range(numberOfTasks):
        G_adjaList, C_dict , T, W = load_task(inputDataFolder, id)
        outputILPSystemJSON(G_adjaList, 0, C_dict, W, numCores, id, dag_file)

def getMaxNeighbours(adjaList):
    max_in = 0
    max_out = 0

    for vertex in adjaList:
        inNeighbours = len(adjaList[vertex]["in"])
        outNeighbours = len(adjaList[vertex]["out"])

        if inNeighbours > max_in:
            max_in = inNeighbours
        if outNeighbours > max_out:
            max_out = outNeighbours

    return max_in, max_out

class DataLoader:

    def __init__(self, input_data_folder, ilp_schedules_folder, numCores = 2, maxNodesPerDag = 30):
        self.numberOfTasks = len(os.listdir(input_data_folder)) // 3
        self.dataFolder = input_data_folder
        self.ilpSchedulesFolder = ilp_schedules_folder
        self.maxNodesPerDag = maxNodesPerDag
        self.tasks = []
        self.tasksFilledUp = []
        self.numCores = numCores
        #node features : C_i / W , deg_in, deg_out, is_source_or_sink, is_in_critical_path
        self.taskFeatures = []
        self.dagTasks = []
        self.ilpOutputs = []

        for id in range(self.numberOfTasks):
            G_adjaList, C_dict , T, W = load_task(self.dataFolder, id)
            self.tasks.append({"G": G_adjaList, "C": C_dict, "T": T, "W": W})
            filledUpAdjaList, filledUpWcets = filledUpAdjaListAndWcets(copy.deepcopy(G_adjaList), copy.deepcopy(C_dict), self.maxNodesPerDag)
            self.tasksFilledUp.append({"G": filledUpAdjaList, "C": filledUpWcets, "T": T, "W": W})
            self.dagTasks.append(getDagTask(G_adjaList, C_dict))
            self.addTaskFeatureMatrix(id, G_adjaList, C_dict, W)
            self.addILPoutput(id)

        self.taskFeatures = torch.Tensor(self.taskFeatures)
        self.ilpOutputs = torch.Tensor(self.ilpOutputs)

    def addILPoutput(self, taskID):
        prioList = getOptimalPriorityListFromILPscheduleFile("%s/schedule_dag_%i.json" % (self.ilpSchedulesFolder, taskID))

        if len(prioList) > self.maxNodesPerDag:
            raise RuntimeError("DataLoader: ILP schedule priorirty list has more priorities than what is permitted (%i > max = %i)" % (len(prioList), self.maxNodesPerDag))
        if len(prioList) < self.maxNodesPerDag:
            for i in range(len(prioList), self.maxNodesPerDag):
                prioList.append(self.maxNodesPerDag - 1)
        
        prioMatrix = [[0 for _ in range(len(prioList))] for _ in range(len(prioList))]
        for task in range(len(prioList)):
            for prio in range(len(prioList)):
                if prioList[task] == prio:
                    prioMatrix[task][prio] = 1
                else:
                    prioMatrix[task][prio] = 0
        self.ilpOutputs.append(prioMatrix)

    def addTaskFeatureMatrix(self, id, adjaList, wcets, totalW):
        self.taskFeatures.append([])
        crit_path, crit_length = criticalPath(adjaList, 0, wcets)
        #max_in_neighbours, max_out_neighbours = getMaxNeighbours(adjaList)
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
            

    def train_val_split(self, train_percentage = 0.8, batch_size = 2, return_dags = False):
        train_threshold = int(train_percentage * self.numberOfTasks)
        batches_indices = list(torch_samplers.BatchSampler(torch_samplers.RandomSampler(self.taskFeatures[:train_threshold]), 
                                                           batch_size=batch_size,drop_last=True))
        train_batches = batches_indices
        #print(self.ilpOutputs)
        val_set = (self.tasksFilledUp[train_threshold:], self.taskFeatures[train_threshold:], self.ilpOutputs[train_threshold:])
        if return_dags:
            return train_threshold, len(self.dagTasks), val_set, self.dagTasks[train_threshold:]
        
        return train_batches, val_set
    
    def getTasksWithFixedNumNodes(self, numNodes, epsilon):
        tasks_ids = []
        for id in range(len(self.tasks)):
            if len(self.tasks[id]['G']) <= numNodes + epsilon and len(self.tasks[id]['G']) >= numNodes - epsilon:
                tasks_ids.append(id)

        return self.taskFeatures[tasks_ids], tasks_ids


if __name__ == "__main__":

    data_file = sys.argv[1]
    dag_file = sys.argv[2]
    num_cores = int(sys.argv[3])
    #print(sys.argv[3])
    outputAllILPSystemJSON("../dag_generator/%s/" % (data_file), numCores=num_cores, dag_file=dag_file)
    

    #test makespan calculation
    #data_loader = DataLoader('../dag_generator/datap8n30/', '../LET-LP-Scheduler/dag_tasks_output_schedules', numCores=2, maxNodesPerDag=15)

    ''' msSolver = ms.MakespanSolver(numberOfCores=2)
    task_108 = data_loader.dagTasks[108]
    _, ilp_priolist = torch.max(data_loader.ilpOutputs[108], dim=1)
    ilp_priolist = ilp_priolist.tolist()
    model_priolist = [14, 14, 14, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

    ilpMakespan = msSolver.computeMakespan(ilp_priolist, task_108)
    modelMakespan = msSolver.computeMakespan(model_priolist, task_108)

    print("ilp: ", ilpMakespan, "model: ", modelMakespan) '''

