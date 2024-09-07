#!/usr/env python3

import torch
import makespan_solver as ms
import data_loader as dl
import makespan_loss
import random as rand
import time
import copy


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

        self.linearLayer = torch.nn.Linear(input_dim*2, input_dim)

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
                attention_in = self.INattentionLayer(torch.concat((h_k.unsqueeze(0), h_INneighbours), dim=0))[0][0, :]
                attention_out = self.OUTattentionLayer(torch.concat((h_k.unsqueeze(0), h_OUTneighbours), dim=0))[0][0, :]

                #concatenating the two resulting attentions from in an out neighbours into one vector
                finalAttentionVector = torch.concat((attention_in, attention_out), dim=0)
                H_kplus1[node] = torch.nn.functional.elu(self.linearLayer(finalAttentionVector), alpha=1.0, inplace=False)

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

        self.attentionLayer = AttentionLayer(embedding_dim, embedding_dim)

        self.ff4 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ff5 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ff6 = torch.nn.Linear(embedding_dim, outDim)

    def forward(self, X: torch.Tensor, taskGraphs):
        
        firstLayer = self.relu(self.ff3(self.relu(self.ff2(self.relu(self.ff1(X))))))
        
        secondLayer = self.sig(self.ff6(self.firstGCN(firstLayer, taskGraphs)))


        #print(secondLayer)
        return secondLayer


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
    

def getAccuracy(output, target):
    avg_accu = 0.0
    for b in range(output.shape[0]):
        _, transform_output = torch.max(output[b], dim=1)
        _, transform_target = torch.max(target[b], dim=1)
        
        accu = transform_output - transform_target
        accu = 1.0 - torch.count_nonzero(accu) / len(accu)

        avg_accu += accu
    
    return avg_accu / output.shape[0]

def train_one_epoch(model, epoch_num, batches, dataLoader, optimizer, loss_fn):
    running_loss = 0
    avg_training_accu = 0.0

    for batch in batches:

        #reset gradients
        optimizer.zero_grad()
        
        outputs = model(dataLoader.taskFeatures[batch], [dataLoader.tasksFilledUp[i] for i in batch])
        
        #compute loss, accu and gradients
        loss = loss_fn(outputs, dataLoader.ilpOutputs[batch])
        avg_training_accu += getAccuracy(outputs, dataLoader.ilpOutputs[batch])

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    #return the average training loss for this epoch
    return running_loss / len(batches), avg_training_accu / len(batches)


def evaluateTiming(model, data_loader, listsOfNumberOfTasks, epsilon):
    
    with open("results_time", "w") as result_file:
        for numTasks in listsOfNumberOfTasks:
            data, ids = data_loader.getTasksWithFixedNumNodes(numTasks, epsilon)
            start = time.time_ns()
            _ = model(data, [data_loader.tasksFilledUp[id] for id in ids])
            end = time.time_ns()
            print(len(ids), numTasks)
            result_file.write("%i tasks -- avg %fµs -- %i samples\n" 
                              % (numTasks, (end - start) * 1.0e-3 / (data.shape[0]), (data.shape[0])))

def training_and_eval(model, num_cores):
    EPOCHS = 1
    data_loader = dl.DataLoader('../dag_generator/data/', '../LET-LP-Scheduler/dag_tasks_output_schedules')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    trainDataBatches, (valTasks, valTaskFeatures, valILPoutputs) = data_loader.train_val_split(train_percentage=0.8, batch_size=5)

    loss_fn = makespan_loss.MakespanLoss(num_cores)


    with open("results_lossaccu", "w+") as result_file:
        for epoch_num in range(EPOCHS):
            #train
            model.train(True)
            avg_train_loss, avg_train_accu = train_one_epoch(model, epoch_num, trainDataBatches, data_loader, optimizer, loss_fn)

            #evaluation
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                voutputs = model(valTaskFeatures, valTasks)
                vloss = loss_fn(voutputs, valILPoutputs)
                avg_vloss = vloss / valILPoutputs.shape[0]
                avg_vaccu = getAccuracy(voutputs, valILPoutputs)

            print("epoch %i -- avg train loss/accu: %f/%f, avg validation loss/accu: %f/%f\n" 
                % (epoch_num, avg_train_loss, avg_train_accu, avg_vloss, avg_vaccu))
            result_file.write("epoch %i -- avg train loss/accu: %f/%f, avg validation loss/accu: %f/%f\n" 
                % (epoch_num, avg_train_loss, avg_train_accu, avg_vloss, avg_vaccu))

    evaluateTiming(model, data_loader, [10, 15, 20], 2)
    evaluateMakespan(model, data_loader, data_loader.numCores)

def evaluateMakespan(trained_model, data_loader, numcores):
    scheduler = ms.MakespanSolver(numcores)
    train_threshold, total_size, (valTasks, valTaskFeatures, valILPoutputs), dags = data_loader.train_val_split(train_percentage=0.9, batch_size=5, return_dags=True)

    outputs = trained_model(valTaskFeatures, valTasks)
    print(train_threshold, total_size)
    with open("results_makespan", "w+") as result_file:
        for id in range(outputs.shape[0]):
            if id == 4:
                print("model: ", outputs[id])
                print("ilp: ", valILPoutputs[id])
            _, model_priorities = torch.max(outputs[id], dim=1)
            _, ilp_priorities = torch.max(valILPoutputs[id], dim=1)
            model_priorities = (model_priorities[:len(dags[id])]).tolist()
            ilp_priorities = (ilp_priorities[:len(dags[id])]).tolist()
            print("priorities: ", model_priorities, ilp_priorities)
            makespan_model = scheduler.computeMakespan(ms.IntVector(model_priorities), dags[id])
            makespan_ilp = scheduler.computeMakespan(ms.IntVector(ilp_priorities), dags[id])

            result_file.write("makespan model %i -- makespan ilp %i -- task id: %i\n" % (makespan_model, makespan_ilp, id))
    

if __name__ == "__main__":

    model = GCNAttention(embedding_dim=5)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    training_and_eval(model, 2)

