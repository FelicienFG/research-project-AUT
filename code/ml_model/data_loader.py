#!/usr/env python3
import networkx as nx
import pickle
import os
import makespan_solver as ms
import torch
import torch.utils.data.sampler as torch_samplers
import numpy
import json
import parse

def getOptimalPriorityListFromILPscheduleFile(scheduleFileName):
    priorityList = []
    with open(scheduleFileName, "r") as scheduleFile:
        scheduleJSON = json.load(scheduleFile)
        scheduledTasks = scheduleJSON['TaskInstancesStore']
        priorityList = [0 for _ in range(len(scheduledTasks))]

        priority = 0
        for taskJSON in scheduledTasks:
            subtaskID = parse.parse("task_{}", taskJSON['name'])
            subtaskID = int(subtaskID[0])
            priorityList[subtaskID - 1] = priority
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

#def getDagTask(graph, wcets):
#    """
#    get the list of DagSubtasks from the graph
#    """
#    dagTask = ms.DagSubtaskVector()
#    for vertex_key in wcets:
#        dagSubtask = ms.DagSubtask()
#        dagSubtask.id = vertex_key - 1 
#        dagSubtask.inDependencies = ms.IntList(graph[vertex_key]['in'])
#        for i in range(dagSubtask.inDependencies.size()):
#            dagSubtask.inDependencies[i] -= 1
#        dagSubtask.outDependencies = ms.IntList(graph[vertex_key]['out'])
#        for i in range(dagSubtask.outDependencies.size()):
#            dagSubtask.outDependencies[i] -= 1
#        dagSubtask.wcet = wcets[vertex_key]
#
#        dagTask.append(dagSubtask)
#
#    return dagTask

#def test_makespan_compute():
#    rand.seed = time.time()
#    makespanSolver = ms.MakespanSolver(numberOfCores=4)
#    dataLoader = dl.DataLoader("../dag_generator/data/")
#    task = 4
#    dagTask = getDagTask(dataLoader.tasks[task]['G'], dataLoader.tasks[task]['C'])
#    
#    priorities = ms.IntVector([rand.randint(0, 4) for i in range(dagTask.size())])
#    
#    makespan = makespanSolver.computeMakespan(priorities, dagTask)
#
#    print(makespan)

def outputILPSystemJSON(graph, start, wcets, totalW, numCores, dagTaskID):
    outputJSON = {}
    outputFile = '../LET-LP-Scheduler/dag_tasks_input_files/multicore_system_DAGtask_%i.json' % (dagTaskID)

    #first add the cores
    outputJSON['CoreStore'] = []
    for core in range(numCores):
        outputJSON['CoreStore'].append({"name": ("c%i" % (core)), "speedup": 1})

    #then add the tasks and edges
    outputJSON['TaskStore'] = []
    outputJSON['DependencyStore'] = []
    for node in graph:
        outputJSON['TaskStore'].append(
            {
                "name": ("task_%i" % (node)),
                "initialOffset":0,
                "activationOffset":0,
                "duration":wcets[node],
                "period":totalW,
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
       


class DataLoader:

    def __init__(self, input_data_folder, ilp_schedules_folder, numCores = 2, maxNodesPerDag = 30):
        self.numberOfTasks = len(os.listdir(input_data_folder)) // 3
        self.dataFolder = input_data_folder
        self.ilpSchedulesFolder = ilp_schedules_folder
        self.maxNodesPerDag = maxNodesPerDag
        self.tasks = []
        self.numCores = numCores
        #node features : C_i / W , deg_in, deg_out, is_source_or_sink, is_in_critical_path
        self.taskFeatures = []
        #self.dagTasks = []
        self.ilpOutputs = []

        for id in range(self.numberOfTasks):
            G_adjaList, C_dict , T, W = load_task(self.dataFolder, id)
            #outputILPSystemJSON(G_adjaList, 1, C_dict, W, numCores, id)
            self.tasks.append({"G": G_adjaList, "C": C_dict, "T": T, "W": W})
            #self.dagTasks.append(getDagTask(G_adjaList, C_dict))
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
                prioList.append(0)

            self.ilpOutputs.append(prioList)

    def addTaskFeatureMatrix(self, id, adjaList, wcets, totalW):
        self.taskFeatures.append([])
        crit_path, crit_length = criticalPath(adjaList, 1, wcets)
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
        val_set = (self.taskFeatures[train_threshold:], self.ilpOutputs[train_threshold:])

        return train_batches, val_set
    
if __name__ == "__main__":

    #data = DataLoader("../dag_generator/data/")
    
    pList = getOptimalPriorityListFromILPscheduleFile("../LET-LP-Scheduler/dag_tasks_output_schedules/schedule_dag_2.json")

    print(pList)