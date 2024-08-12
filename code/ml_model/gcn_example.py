#!/usr/env python3

import torch
import makespan_solver as ms
import data_loader as dl
import makespan_loss
import random as rand
import time


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
    

class AttentionLayer(torch.nn.Module):

    def __init__(self, embedding_dim, n_heads = 1):
        """
        embedding dim is the dimension of the feature vectors, i.e., the numbers of feature for each input vector
        """
        super(AttentionLayer, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads)
        self.Wq = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim))
        self.Wk = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim))
        self.Wv = torch.nn.Parameter(torch.rand( embedding_dim, embedding_dim))

    def forward(self, X: torch.Tensor):
        queries = torch.matmul(X, self.Wq)
        keys = torch.matmul(X, self.Wk)
        values = torch.matmul(X, self.Wv)

        attention_output = self.mha(queries, keys, values, need_weights=False)

        return attention_output

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

def attention_test():
    embed_size = 3
    n_heads = 3
    
    attention_layer = AttentionLayer(embed_size, n_heads)

    X = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.],
                      [10. ,11., 12.]])
    
    output = attention_layer(X)

    print(output)
    

def train_one_epoch(model, epoch_num, batches, optimizer, loss_fn):
    running_loss = 0

    for (dagTaskBatch, featureBatch, ilpOutputs) in batches:

        #reset gradients
        optimizer.zero_grad()
        
        outputs = model(featureBatch)

        #compute loss and gradients
        loss = loss_fn(outputs, dagTaskBatch, ilpOutputs)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    #return the average training loss for this epoch
    return running_loss / len(batches)



def training_and_eval(model, num_cores):
    EPOCHS = 10
    optimizer = torch.optim.sgd.SGD(model.parameters(), lr=0.001)
    trainDataBatches = []#TODO
    valDataBatches = []#TODO

    loss_fn = makespan_loss.MakespanLoss(num_cores)


    for epoch_num in range(EPOCHS):

        #train
        model.train(True)
        avg_train_loss = train_one_epoch(model, epoch_num, trainDataBatches, optimizer, loss_fn)

        #evaluation
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        running_vloss = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for (dagTaskBatch, featureBatch, ilpOutputs) in valDataBatches:
                voutputs = model(featureBatch)
                vloss = loss_fn(dagTaskBatch, voutputs, ilpOutputs)
                running_vloss += vloss
        avg_vloss = running_vloss / len(valDataBatches)


        print("epoch %i -- avg train loss: %f, avg validation loss: %f" % (epoch_num, avg_train_loss, avg_vloss))
        


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

def test_makespan_compute():
    rand.seed = time.time()
    makespanSolver = ms.MakespanSolver(numberOfCores=4)
    dataLoader = dl.DataLoader("../dag_generator/data/")
    task = 4
    dagTask = getDagTask(dataLoader.tasks[task]['G'], dataLoader.tasks[task]['C'])
    
    priorities = ms.IntVector([rand.randint(0, 4) for i in range(dagTask.size())])
    
    makespan = makespanSolver.computeMakespan(priorities, dagTask)

    print(makespan)


if __name__ == "__main__":

    attention_test()