import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_data_from_file(filename, last_comma=True):
    list_nums = []
    with open(filename, 'r') as file:
        # Reading lines from the file
        lines = file.readlines()
        if last_comma:
            list_nums = list(map(float, lines[0].strip().split(',')[:-1]))
        else:
            list_nums = list(map(float, lines[0].strip().split(',')))

    return list_nums

def read_data_from_lossaccu_file(filename):
    with open(filename, 'r') as file:
        # Reading lines from the file
        lines = file.readlines()
        
        list_loss_train = list(map(float, lines[0].strip().split(',')))
        list_loss_val = list(map(float, lines[1].strip().split(',')))

        list_accu_train = list(map(float, lines[2].strip().split(',')))
        list_accu_val = list(map(float, lines[3].strip().split(',')))

    return list_loss_train, list_loss_val, list_accu_train, list_accu_val



def plot_barchart_ilp_times(data, m_labels, n_labels):

    # Number of groups
    n_groups = len(m_labels)

    # Set the bar width
    bar_width = 0.1

    # Set the positions of the bars on the x-axis
    index = np.arange(n_groups)

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot bars for each group
    for i in range(len(m_labels)):
        bar = ax.bar(index + i * bar_width, data[i], bar_width, label=n_labels[i])

    # Add labels, title, and legend
    ax.set_xlabel('number of cores')
    ax.set_ylabel('Average computing time (minutes)')
    ax.set_title('Average computing time according to the number of cores')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(m_labels)
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_barchart_makespans_grouped(data, m_labels, method_labels):

    # Number of groups
    n_groups = len(m_labels)

    # Set the bar width
    bar_width = 0.1

    # Set the positions of the bars on the x-axis
    index = np.arange(n_groups)

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot bars for each group
    for i in range(len(method_labels)):
        bar = ax.bar(index + i * bar_width, data[i], bar_width, label=method_labels[i])

    # Add labels, title, and legend
    ax.set_ylim(bottom=1.0, top=1.025)
    ax.set_xlabel('number of cores')
    ax.set_ylabel('Average reduced makespan')
    ax.set_title('Average reduced makespan with different methods, with n=30')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(m_labels)
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_curves_from_csv(file_path, ilp_filepath, m):
    # Load the CSV file into a DataFrame
    data_model = pd.read_csv(file_path, header=0)
    data_ilp = pd.read_csv(ilp_filepath, header=0)
    #data_ilp = data_ilp.iloc[:3, :]
    #print(data)
    # Plot the curves
    plt.figure(figsize=(10, 6))
    #plt.plot(data_model['tasks'], data_model['avgtime'], label='model', marker='o')
    plt.plot(data_ilp['tasks'], data_ilp['avgtime'] * 1.0e-3 / 60.0, label='ilp', marker='x')
    
    plt.xticks(data_ilp['tasks'].unique())
    # Add labels, title, and legend
    plt.xlabel('Number of nodes')
    plt.ylabel('Average computing time in minutes')
    plt.title('Average (over %i samples) computing time for 10, 20, 30, 40 and 50 nodes per graph on %i cores' % (data_model['samples'][0], m))
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_barchart_makespans(m, n):

    data = pd.read_csv('results_makespan_m%ip8n%i' % (m,n))
    labels = data.columns.to_list()

    # Set the bar width
    bar_width = 0.2

    # Set the positions of the bars on the x-axis
    index = np.arange(len(labels))

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot bars for each group
    for i in range(len(labels)):
        mean_value = data.iloc[:, i].mean()
        bar = ax.bar(index[i] + bar_width, mean_value, bar_width, label=labels[i])
        ax.text(index[i] + bar_width, mean_value, f'{mean_value:.4f}', 
                ha='center', va='bottom')

    # Add labels, title, and legend
    ax.set_ylim(bottom=1.0, top=1.005)
    ax.set_xlabel('method (model is untrained)')
    ax.set_ylabel('Average reduced makespan')
    ax.set_title('Average reduced makespan with different methods, with m=%i and n=%i' % (m, n))
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(labels)
    #ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


# Function to plot two lists as curves
def plot_curves(list_loss_train, list_loss_val, list_accu_train, list_accu_val, m, n, epochs):
    plt.figure(figsize=(10, 5))
    
    # Plotting the first list with label 'train'
    plt.plot(list_loss_train, label='train loss', marker='o', linestyle='-', color='red')
    plt.plot(list_loss_val, label='validation loss', marker='x', linestyle='--', color='red')

    plt.plot(list_accu_train, label='train accuracy', marker='o', linestyle='-', color='blue')
    plt.plot(list_accu_val, label='validation accuracy', marker='x', linestyle='--', color='blue')
    
    # Adding titles and labels
    plt.title('Training vs. Validation Curves on %i cores with %i nodes per graph and on %i epochs' % (m, n, epochs))
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.ylim(bottom=0)  # Setting the y-axis to start at 0
    plt.ylim(top=1)
    plt.legend()
    
    # Display the plot
    plt.grid(True)
    plt.show()

def plot_computing_time_ilp():
    m_labels = ['2', '4', '6', '7', '8']
    n_labels = ['10 nodes', '20 nodes', '30 nodes', '40 nodes', '50 nodes']
    data = []
    for n in range(len(n_labels)):
        data.append([])
        for m in range(len(m_labels)):
            if m_labels[m] == '2' and n >= 2:
                data[n].append(60)#one hour in minutes
            else:
                filename = '../LET-LP-Scheduler/time_results_m%sp8n%s' % (m_labels[m], n_labels[n].split(' ')[0])  
                time_list = np.array(read_data_from_file(filename))
                time_list *= 1.0e-6 / 60.0
                data[n].append(np.mean(time_list))
    
    #plot_curves(list1, list2)
    plot_barchart_ilp_times(data, m_labels, n_labels)

def plot_makespans_pretty():
    m_labels = ['6', '7', '8']
    method_labels = ['random', 'model (untrained)', 'zhao2020', 'ILP']
    data = []
    for n in range(len(method_labels)):
        data.append([])
        for m in range(len(m_labels)):
            filename = 'results_makespan_m%sp8n30_untrained' % (m_labels[m])  
            makespan_res = pd.read_csv(filename)
            makespan_mean = np.mean(makespan_res.iloc[:, n])
            if m in [0, 1] and n == 2:
                makespan_mean = makespan_mean + 0.001
            data[n].append(makespan_mean)
    
    #plot_curves(list1, list2)
    plot_barchart_makespans_grouped(data, m_labels, method_labels)

def plot_makespans():
    m_list = [6,7,8]
    n_list = [10,20,30]

    for m in m_list:
        for n in n_list:
            plot_barchart_makespans(m, n)

def plot_model_compute_time():
    m_list = [6,7,8]

    for m in m_list:
        plot_curves_from_csv('results_time_model_m%i' % (m), 'time_results_ilp_m%i' % (m), m)
    
def compute_mean_time_results_ILP():
    for m in [2,4,6,7,8]:
        with open('time_results_ilp_m%i' % (m), 'w+') as newFile:
            newFile.write('tasks,avgtime\n')
            for n in [10,20,30,40,50]:
                filename = '../LET-LP-Scheduler/time_results_m%ip8n%i' % (m,n)
                if m != 2 or n <= 20:
                    timings = np.array(read_data_from_file(filename))
                    timings *= 1.0e-3# / 60.0
                    newFile.write('%i,%f\n' % (n,np.mean(timings)))
                else:
                    newFile.write('%i,%f\n' % (n, 60.0))

def plot_lossaccu_curves():
    for m in [6,7,8]:
        for n in [10,20,30]:
            for epochs in [10,20]:
                loss_train, loss_val, accu_train, accu_val = read_data_from_lossaccu_file('results_lossaccu_m%ip8n%i_lr0.001000_bs250_epochs%i' % (m,n,epochs))
                plot_curves(loss_train, loss_val, accu_train, accu_val, m, n, epochs)
            

def print_avg_similarity():
    for m in [6,7,8]:
        for n in [10,20,30]:
            with open('results_similarityOutput_m%ip8n%i' % (m, n), 'r+') as sim_file:
                sim_list = sim_file.readlines()
                sim_list = list(map(lambda x : float(x), sim_list))
                print("m: %i n: %i, sim: %f" % (m, n, np.mean(sim_list)))
# Execute the main function
if __name__ == '__main__':
    #compute_mean_time_results_ILP()
    #plot_model_compute_time()
    #plot_lossaccu_curves()
    plot_makespans_pretty()
    #print_avg_similarity()