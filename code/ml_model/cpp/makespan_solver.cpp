#include "makespan_solver.h"
#include <queue>
#include <numeric>
#include <algorithm>
#include "proc_core.h"

static auto dagSubtaskCompare = [](const DagSubtask& left, const DagSubtask& right)
{
    return left.priority < right.priority;
};

typedef std::priority_queue<DagSubtask, std::vector<DagSubtask>, decltype(dagSubtaskCompare)> dagReadyQueue;

int minWorkload(const std::vector<ProcCore>& cores)
{
    int min_w = cores[0].getWorkload();
    for(unsigned i = 1;i<cores.size();++i)
        min_w = std::min(min_w, cores[i].getWorkload());

    return min_w;
}


int maxWorkload(const std::vector<ProcCore>& cores)
{
    int max_w = cores[0].getWorkload();
    for(unsigned i = 1;i<cores.size();++i)
        max_w = std::max(max_w, cores[i].getWorkload());

    return max_w;
}

void updateWaitingListDependencies(std::vector<DagSubtask>& waitingList)
{
    
}

void updateWaitingAndReadyList(std::vector<DagSubtask>& waitingList, dagReadyQueue& readyList)
{
    //push ready tasks in ready list
    //and remove ready tasks from waiting list 
    auto new_end = std::remove_if(waitingList.begin(), waitingList.end(),
    [&readyList](const DagSubtask& task)
    {
        if(task.dependencies.empty())
        {
            readyList.push(task);
            return true;
        }

        return false;
    });

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
    std::vector<DagSubtask>& waitingList = dagTask;
    dagReadyQueue readyList(dagSubtaskCompare);
    std::vector<ProcCore> cores;
    int timer = 0;
    for(int m = 0; m < numCores;++m)
        cores.push_back(ProcCore());

    while(!readyList.empty() || !waitingList.empty())
    {
        //first let's try to assign tasks to processors

        //updating waiting list and ready queue
        updateWaitingListDependencies(waitingList);
        updateWaitingAndReadyList(waitingList, readyList);
    }

    return timer + maxWorkload(cores);
}