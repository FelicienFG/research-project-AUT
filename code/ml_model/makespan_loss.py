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

        #applying cross-entropy
        accu_loss = torch.nn.functional.binary_cross_entropy(output, target_priorities)

        return accu_loss



