#ifndef MAKESPAN_SOLVER_H
#define MAKESPAN_SOLVER_H
#include <vector>

class MakespanSolver
{
public:

    int computeMakespan(const std::vector<int>& priorityList, const std::vector<int>& wcets);
};


#endif //MAKESPAN_SOLVER_H