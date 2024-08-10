#ifndef PROC_CORE_H
#define PROC_CORE_H

class DagSubtask;

class ProcCore
{
public:
    ProcCore(int _id = 0);
    ProcCore(DagSubtask* task);

    DagSubtask* getExecutingTask() const;

    //returns nullptr or the task that was assigned to this core before 
    DagSubtask* assignTask(DagSubtask* newTask, int timer);

    //returns 0 if no task is assigned, wcet of task otherwise.
    int getWorkload(int timer) const;

private:
    DagSubtask* taskExecuting;
    int taskExecuteStartTime;
    int id;
};


#endif //PROC_CORE_H