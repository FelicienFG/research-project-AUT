import torch
import makespan_solver as ms


class MakespanLoss(torch.nn.Module):
    
    def __init__(self, num_cores):
        super(MakespanLoss, self).__init__()

        self.makespanSolver = ms.MakespanSolver(num_cores)

    def forward(self, dagTask, output, target_makespan):
        """
        the output is assumed to be a matrix of scores with the subtasks' ids as rows and priorities as columns
        the target is assumed to be the ILP optimal calculation of the makespan
        """

        _, prio_list = torch.max(output, dim=1)
        makespan = self.makespanSolver.computeMakespan(ms.IntList(prio_list), dagTask)
        
        return 1. - makespan / target_makespan



