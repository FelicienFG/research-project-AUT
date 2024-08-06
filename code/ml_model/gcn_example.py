#!/usr/env python3

import torch
import makespan_solver as ms

make = ms.MakespanSolver()

print(make.computeMakespan([], []))


class myGCNmodule(torch.nn.Module):
    """
        GCN layer

        Args:
            input_dim (int): Dimension of the input
            output_dim (int): Dimension of the output (a softmax distribution)
            A (torch.Tensor): 2D adjacency matrix
    """
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(myGCNmodule, self).__init__()
        self.A = A
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.A_hat = A + torch.eye(self.A.size(0))

        self.ones = torch.ones(input_dim, input_dim)
        self.D = torch.matmul(self.A_hat, self.ones)
        print(self.D.size())
        self.D = torch.diag(self.D)
        self.D_neg_sqrt = torch.diag(torch.pow(self.D, -0.5))
        print(self.D_neg_sqrt.size())
        self.W = torch.nn.Parameter(torch.rand(input_dim, output_dim))


    def forward(self, X: torch.Tensor):

        support_1 = torch.matmul(self.D_neg_sqrt, torch.matmul(self.A_hat, self.D_neg_sqrt))

        support_2 = torch.matmul(support_1, torch.matmul(X, self.W))

        H = torch.nn.functional.relu(support_2)

        return H
    

if __name__ == "__main__":

    input_dim = 3
    output_dim = 2

    A = torch.tensor([[1., 0., 0.],
                      [0., 1., 1.],
                      [0., 1., 1.]])

    gcn_layer = myGCNmodule(input_dim=input_dim, output_dim=output_dim, A=A)

    X = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
    
    output = gcn_layer(X)

    print(output)

    

    