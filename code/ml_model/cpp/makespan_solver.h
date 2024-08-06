#ifndef MAKESPAN_SOLVER_H
#define MAKESPAN_SOLVER_H
#include <vector>
#include <map>

struct DagSubtask
{
    DagSubtask()
        :id(0),wcet(0),priority(0),dependencies(){}
    DagSubtask(int _id, int _wcet, const std::vector<int>& _dependencies)
        :id(_id), wcet(_wcet), priority(0), dependencies(_dependencies){}
    int id;
    int wcet;
    int priority;
    std::vector<int> dependencies;
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
     * @return the exact makespan when scheduled on a preemptive GFPS
     */
    int computeMakespan(const std::vector<int>& priorityList, std::vector<DagSubtask> dagTask);
};


#endif //MAKESPAN_SOLVER_H