import torch
import makespan_solver as ms


class MakespanLoss(torch.nn.Module):
    
    def __init__(self, num_cores):
        super(MakespanLoss, self).__init__()

        self.makespanSolver = ms.MakespanSolver(num_cores)

    def forward(self, dagTask, output, target_makespan):
        """
        the output is assumed to be a batch of matrices of scores with the subtasks' ids as rows and priorities as columns
        the target is assumed to be the batch of ILP optimal calculations of the makespan
        """

        accu_loss = 0.0
        #print(output)
        for matrix_index in range(output.shape[0]):

            _, prio_list = torch.max(output[matrix_index], dim=1)
            #problem, makespan calculation not differentiable, IDEA: forward pass outputs the schedule instead of just a priority list
            makespan = 1#self.makespanSolver.computeMakespan(ms.IntVector(prio_list.tolist()), dagTask[matrix_index])
            
            accu_loss += 1. - makespan / target_makespan[matrix_index]

        return accu_loss



