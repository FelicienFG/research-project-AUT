#include "proc_core.h"
#include "makespan_solver.h"

ProcCore::ProcCore()
    :taskExecuting(nullptr), taskExecuteStartTime(0)
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
    return (taskExecuting ? std::max(taskExecuting->wcet - (timer - taskExecuteStartTime),0) : 0);
}
