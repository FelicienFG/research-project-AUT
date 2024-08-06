#ifndef PROC_CORE_H
#define PROC_CORE_H

class DagSubtask;

class ProcCore
{
public:
    ProcCore();
    ProcCore(DagSubtask* task);

    DagSubtask* getExecutingTask() const;

    //returns nullptr or the task that was assigned to this core before 
    DagSubtask* assignTask(DagSubtask* newTask);

    //returns 0 if no task is assigned, wcet of task otherwise.
    int getWorkload() const;

private:

    DagSubtask* taskExecuting;
};


#endif //PROC_CORE_H