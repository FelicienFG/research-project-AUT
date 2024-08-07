#include "makespan_solver.h"
#include <queue>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "proc_core.h"

static auto dagSubtaskCompare = [](const DagSubtask* left, const DagSubtask* right)
{
    return left->priority < right->priority;
};

typedef std::priority_queue<DagSubtask*, std::vector<DagSubtask*>, decltype(dagSubtaskCompare)> dagReadyQueue;

int minPosWorkload(const std::vector<ProcCore>& cores, int timer)
{
    int min_w = cores[0].getWorkload(timer);
    for(unsigned i = 1;i<cores.size();++i)
        if(cores[i].getWorkload(timer) > 0)
            min_w = std::min(min_w, cores[i].getWorkload(timer));

    return min_w;
}


int maxWorkload(const std::vector<ProcCore>& cores, int timer)
{
    int max_w = cores[0].getWorkload(timer);
    for(unsigned i = 1;i<cores.size();++i)
        max_w = std::max(max_w, cores[i].getWorkload(timer));

    return max_w;
}

void updateDependencies(std::vector<DagSubtask>& dagTask, const std::vector<ProcCore>& cores, int timer)
{
    for(const ProcCore& core : cores)
    {
        DagSubtask* executingTask = core.getExecutingTask();
        //if the executing task just finished executing
        if(executingTask && core.getWorkload(timer) == 0)
        {
            for(int neighbor : executingTask->outDependencies)
            {
                dagTask[neighbor].inDependencies.remove(executingTask->id);
            }
        }
    }
}

void updateWaitingAndReadyList(std::vector<DagSubtask*>& waitingList, dagReadyQueue& readyList)
{
    if(waitingList.empty())
        return;
    //push ready tasks in ready list
    //and remove ready tasks from waiting list 
    auto new_end = std::remove_if(waitingList.begin(), waitingList.end(),
    [&readyList](DagSubtask* task)
    {

        if(task->inDependencies.empty())
        {
            readyList.push(task);
            return true;
        }
        return false;
    });

    if(new_end == waitingList.end())
        waitingList.clear();
    else
        waitingList.erase(new_end, waitingList.end());
}

MakespanSolver::MakespanSolver(int numberOfCores)
    :numCores(numberOfCores)
{
}

int MakespanSolver::computeMakespan(const std::vector<int> &priorityList, std::vector<DagSubtask> dagTask)
{
    for(unsigned i = 0;i<priorityList.size();++i)
        dagTask[i].priority = priorityList[i];

    //first initialize the different arrays and queue
    std::vector<DagSubtask*> waitingList;
    dagReadyQueue readyList(dagSubtaskCompare);
    std::vector<ProcCore> cores;
    int timer = 0;

    for(int m = 0; m < numCores;++m)
        cores.push_back(ProcCore());
    for(DagSubtask& task : dagTask)
        waitingList.push_back(&task);

    while(!readyList.empty() || !waitingList.empty())
    {
        //first update waiting list and ready queue
        updateDependencies(dagTask, cores, timer);
        updateWaitingAndReadyList(waitingList, readyList);
        //then test if a core can execute the readlist's top task
        
        for(ProcCore& core : cores)
        {
            if(readyList.empty())
                break;
            //either the core is free, or the task executing on it has a lower priority (higher value) than the ready task
            if(core.getWorkload(timer) == 0 || (core.getExecutingTask() && core.getExecutingTask()->priority > readyList.top()->priority))
            {
                core.assignTask(readyList.top(), timer);
                readyList.pop();
            }
        }

        timer += minPosWorkload(cores, timer);
    }

    return timer + maxWorkload(cores, timer) + dagTask[dagTask.size() - 1].wcet;
}