#ifndef MAKESPAN_SOLVER_H
#define MAKESPAN_SOLVER_H
#include <vector>
#include <map>
#include <list>

struct DagSubtask
{
    DagSubtask()
        :id(0),wcet(0),priority(0), inDependencies(), outDependencies(){}
    DagSubtask(int _id, int _wcet, const std::list<int>& _inDependencies, const std::list<int>& _outDependencies)
        :id(_id), wcet(_wcet), priority(0), inDependencies(_inDependencies), outDependencies(_outDependencies){}
    int id;
    int wcet;
    int priority;
    std::list<int> inDependencies;
    std::list<int> outDependencies;
};

class MakespanSolver
{
    int numCores;
public:

    MakespanSolver(int numberOfCores = 4);

    /**
     * @brief computes the exact makespan of a dag task when scheduled using priority-list scheduling
     * on a preemptive Global Fixed-Priority Scheduler (GFPS)
     * 
     * @param priorityList the indices must correspond to the ID of subtasks
     * @param dagTask the indices must correspond to the ID of each dag subtask.
     * @param priorityOrder if equal to 1 then the lower the priority value, the higher the execution priority, if equal
     * to -1, the higher the priority value, the higher the execution priority.
     * @return the exact makespan when scheduled on a preemptive GFPS
     */
    int computeMakespan(const std::vector<int>& priorityList, std::vector<DagSubtask> dagTask, int priorityOrder = 1);

    /**
     * @brief gives the priority list that gives the minimum makespan for a specific DAG task,
     * it uses brute force by looping through each possible permutation of priorities to find the one minimizing the makespan.
     * 
     * @param dagTask the dag task to schedule, the priorities will range from 0 to the total number of subtasks - 1
     * 
     * @return priority list that minimizes the makespan for dagTask using non-preemptive GFPS
     */
    std::vector<int> computeBestPriorityList(const std::vector<DagSubtask>& dagTask);


    //returns every permutations of the priorities vector
    static std::vector<std::list<int>> computePermutations(std::vector<int>& priorities);

};


#endif //MAKESPAN_SOLVER_H