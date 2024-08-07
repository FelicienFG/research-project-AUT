#!/usr/env python3
import networkx as nx
import pickle
import os


class DataLoader:

    def __init__(self, data_folder):
        self.numberOfTasks = len(os.listdir(data_folder)) // 3
        self.dataFolder = data_folder
        self.tasks = []

        for id in range(self.numberOfTasks):
            G_adjaList, C_dict , T, W = self.load_task(id)
            self.tasks.append({"G": G_adjaList, "C": C_dict, "T": T, "W": W})



    def load_task(self, id):
        dag_task_file = self.dataFolder + "Tau_{:d}.gpickle".format(id)

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
    

if __name__ == "__main__":

    data = DataLoader("../dag_generator/data/")
    print(data.tasks)