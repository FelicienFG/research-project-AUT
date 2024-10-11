#!/usr/env python3

import torch
import makespan_solver as ms
import data_loader as dl
import makespan_loss
import random as rand
import time
import copy

#torch.autograd.set_detect_anomaly(True)

class AttentionModule(torch.nn.Module):

    def __init__(self, embedding_dim):
        super(AttentionModule, self).__init__()

        self.W_sum = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim))
        self.W_alpha = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim))
        self.a_alpha = torch.nn.Parameter(torch.rand(embedding_dim, 1))
        self.b_alpha = torch.nn.Parameter(torch.rand(embedding_dim, 1))

    def forward(self, X):
        """
        it is assumed that X[0] is h_i and X[1:] are its neighbours
        """
        alpha_ij = torch.zeros((X.shape[0], 1))
        final_sum = torch.zeros_like(X)
        
        for j in range(X.shape[0]):
            
            a_w = torch.matmul(self.a_alpha.T, self.W_alpha)
            b_w = torch.matmul(self.b_alpha.T, self.W_alpha)
            alpha_exponent = torch.matmul(a_w, X[0].T) + torch.matmul(b_w, X[j].T)
            alpha_ij[j] = torch.exp(alpha_exponent)
            final_sum[j] = alpha_ij[j].clone() * X[j]
        
        sum_output = torch.div(torch.sum(final_sum, dim=0), torch.sum(alpha_ij))
        final_output = torch.matmul(sum_output, self.W_sum)

        return final_output

class AttentionLayer(torch.nn.Module):

    def __init__(self, embedding_dim, n_heads = 2):
        """
        embedding dim is the dimension of the feature vectors, i.e., the numbers of feature for each input vector
        """
        super(AttentionLayer, self).__init__()
        
        self.elu = torch.nn.ELU()
        self.linear = torch.nn.Linear(embedding_dim * n_heads, embedding_dim)

        self.inAttention = AttentionModule(embedding_dim)
        self.outAttention = AttentionModule(embedding_dim)

    def forward(self, X_in: torch.Tensor, X_out: torch.Tensor):

        attentionINresult = self.inAttention(X_in)
        attentionOUTresult = self.outAttention(X_out)

        attention_output = torch.concat((attentionINresult, attentionOUTresult), dim=0)
        attention_output = self.elu(self.linear(attention_output))

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
        self.attentionLayer = AttentionLayer(input_dim)

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
                attention_in = torch.concat((h_k.unsqueeze(0), h_INneighbours), dim=0)
                attention_out = torch.concat((h_k.unsqueeze(0), h_OUTneighbours), dim=0)
                
                H_kplus1[node] = self.attentionLayer(attention_in, attention_out)

            H_kplus1_batch[batch] = H_kplus1

        return H_kplus1_batch
    
class GCNAttention(torch.nn.Module):

    def __init__(self, embedding_dim, outDim = 30):
        super(GCNAttention, self).__init__()

        self.ff1 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ff2 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ff3 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Tanh()

        self.firstGCN  = myGCNmodule(embedding_dim)
        self.secondGCN  = myGCNmodule(embedding_dim)
        self.thirdGCN  = myGCNmodule(embedding_dim)

        self.ff4 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ff5 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ff6 = torch.nn.Linear(embedding_dim, outDim)
        self.printForward = False

    def forward(self, X: torch.Tensor, taskGraphs):
        
        if self.printForward:
            print("initial: ", X[0])
        firstLayer = self.relu(self.ff1(X))

        if self.printForward:
            print("first: ", firstLayer[0])
        secondLayer = self.relu(self.ff2(firstLayer))
        
        if self.printForward:
           print("second: ", secondLayer[0])
        thirdLayer = self.relu(self.ff3(secondLayer))
        
        if self.printForward:
           print("third: ", thirdLayer[0])
        fourthLayer = self.firstGCN(X, taskGraphs)

        if self.printForward:
           print("fourth: ", fourthLayer[0])
        fifthLayer = self.relu(self.ff4(fourthLayer))

        if self.printForward:
           print("fifth: ", fifthLayer[0])
        sixthLayer = self.relu(self.ff5(fifthLayer))

        if self.printForward:
           print("sixth: ", sixthLayer[0])
        finalLayer = self.sig(self.ff6(fifthLayer))

        if self.printForward:
           print("final: ", finalLayer[0])
        return finalLayer


    

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


def evaluateTiming(model, numCores, listsOfNumberOfTasks, maxTasks=100):
    
    p=8
    m=numCores
    with open("results_time", "w") as result_file:
        result_file.write("tasks, avgtime, samples\n")
        for n in listsOfNumberOfTasks:
            data_loader = dl.DataLoader('../dag_generator/data_p%in%i' % (p, n), '../LET-LP-Scheduler/dag_m%ip%in%i_output_schedules' % (m,p,n), numCores, n, maxTasks)
            data, ids = data_loader.getTasksWithFixedNumNodes(n, 0)
            start = time.time_ns()
            _ = model(data, [data_loader.tasksFilledUp[id] for id in ids])
            end = time.time_ns()
            print("eval timing for ", n, " tasks")
            result_file.write("%i, %f, %i\n" 
                              % (n, (end - start) * 1.0e-3 / (data.shape[0]), (data.shape[0])))

def write_to_csv(opened_file, input_list):
    for i in range(len(input_list) - 1):
        opened_file.write("%f, " % (input_list[i]))
    opened_file.write("%f\n" % (input_list[len(input_list) - 1]))

def training_and_eval(num_cores, numTasks, p=8, learn_rate=0.001, batch_size=250, epochs=10):
    model = GCNAttention(embedding_dim=5, outDim=15)
    EPOCHS = epochs
    data_loader = dl.DataLoader('../dag_generator/data_p%in%i/' % (p, numTasks), '../LET-LP-Scheduler/dag_tasks_output_schedules', numCores=2, maxNodesPerDag=15, maxTasks=1400)
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    #0.7145 to have 
    trainDataBatches, (valTasks, valTaskFeatures, valILPoutputs) = data_loader.train_val_split(train_percentage=0.7145, batch_size=batch_size)

    loss_fn = makespan_loss.MakespanLoss(num_cores)
    train_loss_scores = []
    val_loss_scores = []
    train_accu_scores = []
    val_accu_scores = []

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
        train_loss_scores.append(avg_train_loss)
        train_accu_scores.append(avg_train_accu)
        val_loss_scores.append(avg_vloss)
        val_accu_scores.append(avg_vaccu)
                    
    with open("results_lossaccu_m%ip%in%i_lr%f_bs%i_epochs%i" % (num_cores, p, numTasks, learn_rate, batch_size, epochs), "w+") as result_file:
        write_to_csv(result_file, train_loss_scores)
        write_to_csv(result_file, val_loss_scores)
        write_to_csv(result_file, train_accu_scores)
        write_to_csv(result_file, val_accu_scores)

    torch.save(model.state_dict(), 'model_weights_m%ip%in%i_lr%f_bs%i_epochs%i.pth' % (num_cores, p, numTasks, learn_rate, batch_size, epochs))

def evaluateMakespan(trained_model, data_loader, numcores):
    scheduler = ms.MakespanSolver(numcores)
    train_threshold, total_size, (valTasks, valTaskFeatures, valILPoutputs), dags = data_loader.train_val_split(train_percentage=0.9, batch_size=5, return_dags=True)

    outputs = trained_model(valTaskFeatures, valTasks)
    print(train_threshold, total_size)
    with open("results_makespan", "w+") as result_file:
        result_file.write("model, ilp\n")
        for id in range(outputs.shape[0]):
            #extract priority lists
            _, model_priorities = torch.max(outputs[id], dim=1)
            _, ilp_priorities = torch.max(valILPoutputs[id], dim=1)
            #drop out the excess priorities from dummy nodes added beforehand
            model_priorities = (model_priorities[:len(dags[id])]).tolist()
            ilp_priorities = (ilp_priorities[:len(dags[id])]).tolist()
            print("priorities: ", model_priorities, ilp_priorities)
            #compute makespans
            makespan_model = scheduler.computeMakespan(ms.IntVector(model_priorities), dags[id])
            makespan_ilp = scheduler.computeMakespan(ms.IntVector(ilp_priorities), dags[id])

            result_file.write("%i, %i\n" % (makespan_model, makespan_ilp))
    

if __name__ == "__main__":

    for n in [10,20]:
        for lr in [10**(-i) for i in range(1, 10)]:
            for bs in [10*x for x in range(1, 31)]:
                training_and_eval(2, n, p=8, learn_rate=lr, batch_size=bs)

    for m in [4,6,7,8]:
        for n in [10,20,30]:
            for lr in [10**(-i) for i in range(1, 10)]:
                for bs in [10*x for x in range(1, 31)]:
                    training_and_eval(m, n, p=8, learn_rate=lr, batch_size=bs)

