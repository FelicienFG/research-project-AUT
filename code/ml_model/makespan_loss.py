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
        accu_loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target_priorities)
        reg_term = 0.0
        for X in output:
            avg_dist = 0.0
            for node_a in range(X.shape[0]):
                for node_b in range(node_a+1, X.shape[0]):
                    avg_dist += torch.dist(X[node_a], X[node_b], p=2)

            avg_dist /= X.shape[0]
            reg_term += avg_dist

        return accu_loss - reg_term



