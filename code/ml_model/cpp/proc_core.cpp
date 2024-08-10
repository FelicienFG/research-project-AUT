#include "proc_core.h"
#include "makespan_solver.h"
#include <iostream>

ProcCore::ProcCore(int _id)
    :taskExecuting(nullptr), taskExecuteStartTime(0), id(_id)
{
}

ProcCore::ProcCore(DagSubtask *task)
    :taskExecuting(task), taskExecuteStartTime(0)
{
}

DagSubtask *ProcCore::getExecutingTask() const
{
    return taskExecuting;
}

DagSubtask *ProcCore::assignTask(DagSubtask *newTask, int timer)
{
    DagSubtask* taskBuffer = taskExecuting;
    taskExecuting = newTask;
    taskExecuteStartTime = timer;

    return taskBuffer;
}

int ProcCore::getWorkload(int timer) const
{
    /* if(taskExecuting)
        std::cout<<"core: "<<id<<" wl: "<< taskExecuting->wcet - (timer - taskExecuteStartTime)<<std::endl; */
    return (taskExecuting ? std::max(taskExecuting->wcet - (timer - taskExecuteStartTime),0) : 0);
}
