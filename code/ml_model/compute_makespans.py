import makespan_solver as ms
import data_loader as dl
from rta_alphabeta_new import load_task
from rta_alphabeta_new import Eligiblity_Ordering_PA, TPDS_Ordering_PA
import os
import pandas as pd


def load_all_tasks(inputDataFolder):
    numberOfTasks = len(os.listdir(inputDataFolder)) // 3
    listOfDags = []
    listOfCustomDags = []

    for id in range(numberOfTasks):
        G_adjaList, C_dict, _, _, _ , _, _ = load_task(inputDataFolder, id)
        G_adjaList_custom, C_dict_custom, _, _ = dl.load_task(inputDataFolder, id)
        listOfDags.append({"G": G_adjaList, "C": C_dict})
        listOfCustomDags.append({"G": G_adjaList_custom, "C": C_dict_custom})

    return listOfDags, listOfCustomDags


def convertDictPrioritiesToIntVector(prios):
    prioList = []
    for key in sorted(prios.keys()):
        prioList.append(prios[key])
    
    return ms.IntVector(prioList)

if __name__ == "__main__":

    numCores = 2
    msSolver = ms.MakespanSolver(numCores)
    #dags custom have in and out neighbours in the adjacency list
    dags, dags_custom = load_all_tasks('../dag_generator/data/')
    dagsVector = list(map(lambda dag: dl.getDagTask(dag['G'], dag['C']), dags_custom))
    makespans = {
        "zhao2020" : [],
        "he2019": []
    }

    for i in range(len(dagsVector)):
        prio_zhao2020 = convertDictPrioritiesToIntVector(Eligiblity_Ordering_PA(dags[i]['G'], dags[i]['C']))
        prio_he2019 = convertDictPrioritiesToIntVector(TPDS_Ordering_PA(dags[i]['G'], dags[i]['C']))
        
        makespan_zhao2020 = msSolver.computeMakespan(prio_zhao2020, dagsVector[i], -1)
        makespan_he2019 = msSolver.computeMakespan(prio_he2019, dagsVector[i])

        makespans["zhao2020"].append(makespan_zhao2020)
        makespans["he2019"].append(makespan_he2019)

    makespans = pd.DataFrame(makespans)
    print(makespans.describe())
