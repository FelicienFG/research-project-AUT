#include "makespan_solver.h"
#include <queue>
#include <limits>
#include <algorithm>
#include <iostream>
#include "proc_core.h"

namespace
{
    size_t factorial(size_t num)
    {
        if(num <= 1)
            return 1;
        
        return num * factorial(num - 1);
    }
}

class dagSubtaskCompare 
{
public:
    int priorityOrder;

    dagSubtaskCompare(int _priorityOrder = 1)
        :priorityOrder(_priorityOrder){}

    bool operator()(const DagSubtask* left, const DagSubtask* right)
    {
        if(priorityOrder == -1)
            return left->priority < right->priority;
        
        return left->priority > right->priority;
         
    }
};

typedef std::priority_queue<DagSubtask*, std::vector<DagSubtask*>, dagSubtaskCompare> dagReadyQueue;

int minPosWorkload(const std::vector<ProcCore>& cores, int timer)
{
    int min_w = std::numeric_limits<int>::max();
    for(unsigned i = 0;i<cores.size();++i)
        if(cores[i].getWorkload(timer) > 0)
            min_w = std::min(min_w, cores[i].getWorkload(timer));
    if(min_w >= std::numeric_limits<int>::max())
        return 1;
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

    if(new_end != waitingList.end())
        waitingList.erase(new_end, waitingList.end());
}

MakespanSolver::MakespanSolver(int numberOfCores)
    :numCores(numberOfCores)
{
}

void printreadlist(dagReadyQueue list)
{
    while(!list.empty())
    {
        std::cout<<list.top()->id<<" ";
        list.pop();
    }
}

void printwaitinglist(const std::vector<DagSubtask*>& list)
{
    for(DagSubtask* task : list)
        std::cout<<task->id<<" ";
}

int MakespanSolver::computeMakespan(const std::vector<int> &priorityList, std::vector<DagSubtask> dagTask, int priorityOrder)
{
    for(unsigned i = 0;i<priorityList.size();++i)
        dagTask[i].priority = priorityList[i];

    //first initialize the different arrays and queue
    std::vector<DagSubtask*> waitingList;
    dagReadyQueue readyList{dagSubtaskCompare(priorityOrder)};
        
    std::vector<ProcCore> cores;
    int timer = 0;

    for(int m = 0; m < numCores;++m)
        cores.push_back(ProcCore(m));
    for(DagSubtask& task : dagTask)
        waitingList.push_back(&task);

    while(!readyList.empty() || !waitingList.empty())
    {
        //first update waiting list and ready queue
/*         std::cout<<"before ready: ";
        printreadlist(readyList);
        std::cout<<std::endl<<"before waiting: ";
        printwaitinglist(waitingList);
        std::cout<<std::endl; */
        updateDependencies(dagTask, cores, timer);
        updateWaitingAndReadyList(waitingList, readyList);
        //then test if a core can execute the readlist's top task
        /* std::cout<<"after ready: ";
        printreadlist(readyList);
        std::cout<<std::endl<<"after waiting: ";
        printwaitinglist(waitingList);
        std::cout<<std::endl; */

        for(ProcCore& core : cores)
        {
            if(readyList.empty())
                break;
            //if the core is free
            if(core.getWorkload(timer) == 0)
            {
                core.assignTask(readyList.top(), timer);
                readyList.pop();
            }
        }

        timer += minPosWorkload(cores, timer);
    }

    return timer + maxWorkload(cores, timer);
}

std::vector<std::list<int>> MakespanSolver::computePermutations(std::vector<int>& priorities)
{
    if(priorities.size() == 2)
        return {{priorities[0], priorities[1]}, {priorities[1], priorities[0]}};
    
    //compute the permutations of the priorities vector minus the last element
    //and then for each permutation, insert the last element in every spot to compute all the permutations
    int lastItem = priorities.back();
    priorities.pop_back();
    std::vector<std::list<int>> permutationsMinusOne = computePermutations(priorities);
    std::vector<std::list<int>> permutations;

    for(auto& permutation : permutationsMinusOne)
    {
        for(auto iter = permutation.begin();iter!=permutation.end();++iter)
        {
            auto lastInsertedPos = permutation.insert(iter, lastItem);
            permutations.push_back(permutation);
            permutation.erase(lastInsertedPos);
        }

        auto lastInsertedPos = permutation.insert(permutation.end(), lastItem);
        permutations.push_back(permutation);
        permutation.erase(lastInsertedPos);
    }

    return permutations;
}

void initVectorFromList(std::vector<int>& vector, std::list<int>& list)
{
    size_t i = 0;
    for(const int& element : list)
    {
        vector[i++] = element;
    }
}

std::vector<int> MakespanSolver::computeBestPriorityList(const std::vector<DagSubtask> &dagTask)
{
    size_t maxPriority = dagTask.size();
    std::vector<int> priorities(maxPriority);
    std::vector<int> minimumPriorities(maxPriority);
    int minimumMakespan = std::numeric_limits<int>::max();
    int currentMakespan = 0;

    for(size_t p = 0; p < maxPriority; ++p)
        priorities[p] = p;

    std::vector<std::list<int>> priorityPermutations = computePermutations(priorities);

    for(std::list<int>& priorityPermutation : priorityPermutations)
    {
        std::vector<int> prioritiesVector(priorityPermutation.size());
        initVectorFromList(prioritiesVector, priorityPermutation);
        currentMakespan = computeMakespan(prioritiesVector, dagTask);
        if(currentMakespan < minimumMakespan)
        {
            minimumMakespan = currentMakespan;
            minimumPriorities = prioritiesVector;
        }

    }

    return minimumPriorities;
}
