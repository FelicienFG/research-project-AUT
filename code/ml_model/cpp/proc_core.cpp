#include "proc_core.h"
#include "makespan_solver.h"

ProcCore::ProcCore()
    :taskExecuting(nullptr)
{
}

ProcCore::ProcCore(DagSubtask *task)
    :taskExecuting(task)
{
}

DagSubtask *ProcCore::getExecutingTask() const
{
    return taskExecuting;
}

DagSubtask *ProcCore::assignTask(DagSubtask *newTask)
{
    DagSubtask* taskBuffer = taskExecuting;
    taskExecuting = newTask;

    return taskBuffer;
}

int ProcCore::getWorkload() const
{
    return (taskExecuting ? taskExecuting->wcet : 0);
}
