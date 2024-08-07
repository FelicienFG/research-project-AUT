#!/usr/env python3

import torch
import makespan_solver as ms
import data_loader as dl


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
    

def gcn_test():

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

def getDagTask(graph, wcets):
    """
    get the list of DagSubtasks from the graph
    """
    dagTask = ms.DagSubtaskVector()
    for vertex_key in wcets:
        dagSubtask = ms.DagSubtask()
        dagSubtask.id = vertex_key - 1 
        dagSubtask.inDependencies = ms.IntList(graph[vertex_key]['in'])
        for i in range(dagSubtask.inDependencies.size()):
            dagSubtask.inDependencies[i] -= 1
        dagSubtask.outDependencies = ms.IntList(graph[vertex_key]['out'])
        for i in range(dagSubtask.outDependencies.size()):
            dagSubtask.outDependencies[i] -= 1
        dagSubtask.wcet = wcets[vertex_key]

        dagTask.append(dagSubtask)

    return dagTask


if __name__ == "__main__":

    makespanSolver = ms.MakespanSolver()
    dataLoader = dl.DataLoader("../dag_generator/data/")
    task = 1
    dagTask = getDagTask(dataLoader.tasks[task]['G'], dataLoader.tasks[task]['C'])
    
    priorities = ms.IntVector([0, 1, 1, 1, 2, 3])
    
    makespan = makespanSolver.computeMakespan(priorities, dagTask)

    print(makespan)