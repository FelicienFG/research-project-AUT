import torch
#import makespan_solver as ms


class MakespanLoss(torch.nn.Module):
    
    def __init__(self, num_cores):
        super(MakespanLoss, self).__init__()

        #self.makespanSolver = ms.MakespanSolver(num_cores)

    def forward(self, output, target_priorities):
        """
        the output is assumed to be a batch of matrices of scores with the subtasks' ids as rows and priorities as columns
        the target is assumed to be the batch of ILP optimal calculations of the priorities
        """

        accu_loss = 0.0
        #print(output)
        #for matrix_index in range(output.shape[0]):

        _, prio_list = torch.max(output, dim=0)
        #applying rmse
        accu_loss += torch.sqrt(torch.sum(torch.square(target_priorities[0] - prio_list)) / len(prio_list))

        return accu_loss



