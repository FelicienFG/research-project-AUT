#!/usr/env python3

import torch
import makespan_solver as ms
import data_loader as dl
import makespan_loss
import random as rand
import time


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

class myGCNmodule(torch.nn.Module):
    """
        GCN layer

        Args:
            input_dim (int): Dimension of the input
            output_dim (int): Dimension of the output (a softmax distribution)
            A (torch.Tensor): 2D adjacency matrix
    """
    def __init__(self, input_dim: int):
        super(myGCNmodule, self).__init__()
        self.INattentionLayer = AttentionLayer(input_dim, n_heads=5)
        self.OUTattentionLayer = AttentionLayer(input_dim, n_heads=5)
        self.FINALattentionLayer = torch.nn.MultiheadAttention(input_dim, num_heads=1)
        self.linearLayer = torch.nn.Linear(input_dim, input_dim)

    def forward(self, X_batch: torch.Tensor, taskGraph_batch):
        H_kplus1_batch = torch.zeros_like(X_batch)
        for batch in range(X_batch.shape[0]):
            X = X_batch[batch]
            taskGraph = taskGraph_batch[batch]        
            H_kplus1 = torch.zeros_like(X)
            for node in range(X.shape[0]):
                h_k = X[node]
                #get the neighbours
                h_INneighbours = X[taskGraph['G'][node]['in'],:]
                h_OUTneighbours = X[taskGraph['G'][node]['out'],:]
                #attention with the in and out neighbours
                attention_in = self.INattentionLayer(torch.concat((h_k.unsqueeze(0), h_INneighbours), dim=0))[0]
                attention_out = self.OUTattentionLayer(torch.concat((h_k.unsqueeze(0), h_OUTneighbours), dim=0))[0]
                attention = torch.concat((attention_in, attention_out), dim=0)

                #self attention with the two resulting attentions from in an out neighbours
                finalAttention, _ = self.FINALattentionLayer(attention, attention, attention, need_weights=False)
                finalAttentionVector = torch.max(finalAttention, dim=0)[0]
                output = torch.nn.functional.elu(self.linearLayer(finalAttentionVector), alpha=1.0)
                H_kplus1[node] = output

            H_kplus1_batch[batch] = H_kplus1

        return H_kplus1_batch
    
class GCNAttention(torch.nn.Module):

    def __init__(self, embedding_dim, outDim = 30):
        super(GCNAttention, self).__init__()

        self.ff1 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ff2 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ff3 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()

        self.firstGCN  = myGCNmodule(embedding_dim)
        self.secondGCN = myGCNmodule(embedding_dim)
        self.thirdGCN = myGCNmodule(embedding_dim)

        self.attentionLayer = AttentionLayer(embedding_dim, embedding_dim)

        self.ff4 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ff5 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ff6 = torch.nn.Linear(embedding_dim, outDim)

    def forward(self, X: torch.Tensor, taskGraphs):
        
        firstLayer = self.relu(self.ff3(self.relu(self.ff2(self.relu(self.ff1(X))))))

        secondLayer = self.thirdGCN(self.secondGCN(self.firstGCN(firstLayer, taskGraphs), taskGraphs), taskGraphs)

        thirdLayer, _ = self.attentionLayer(secondLayer)
    
        finalLayer = self.sig(self.ff6(self.relu(self.ff5(self.relu(self.ff4(thirdLayer))))))

        return finalLayer

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
    

def train_one_epoch(model, epoch_num, batches, dataLoader, optimizer, loss_fn):
    running_loss = 0

    for batch in batches:

        #reset gradients
        optimizer.zero_grad()
        
        outputs = model(dataLoader.taskFeatures[batch], [dataLoader.tasksFilledUp[i] for i in batch])
        
        #compute loss and gradients
        loss = loss_fn(outputs, dataLoader.ilpOutputs[batch])
        loss.requires_grad = True
        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    #return the average training loss for this epoch
    return running_loss / len(batches)



def training_and_eval(model, num_cores):
    EPOCHS = 5
    data_loader = dl.DataLoader('../dag_generator/data/', '../LET-LP-Scheduler/dag_tasks_output_schedules')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    trainDataBatches, (valTasks, valTaskFeatures, valILPoutputs) = data_loader.train_val_split(train_percentage=0.7, batch_size=2)

    loss_fn = makespan_loss.MakespanLoss(num_cores)


    for epoch_num in range(EPOCHS):

        #train
        model.train(True)
        avg_train_loss = train_one_epoch(model, epoch_num, trainDataBatches, data_loader, optimizer, loss_fn)

        #evaluation
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        running_vloss = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            voutputs = model(valTaskFeatures, valTasks)
            vloss = loss_fn(voutputs, valILPoutputs)
            avg_vloss = vloss / valILPoutputs.shape[0]


        print("epoch %i -- avg train loss: %f, avg validation loss: %f" % (epoch_num, avg_train_loss, avg_vloss))


if __name__ == "__main__":

    model = GCNAttention(embedding_dim=5)
    training_and_eval(model, 2)
