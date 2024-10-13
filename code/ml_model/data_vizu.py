import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to read data from file and return two lists of values
def read_data_from_file(filename):
    with open(filename, 'r') as file:
        # Reading lines from the file
        lines = file.readlines()

        list_nums = list(map(float, lines[0].strip().split(',')[:-1]))

    return list_nums
    
def plot_barchart(data, m_labels, n_labels):

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



# Function to plot two lists as curves
def plot_curves(list1, list2):
    plt.figure(figsize=(10, 5))
    
    # Plotting the first list with label 'train'
    plt.plot(list1, label='training', marker='o', linestyle='-')
    
    # Plotting the second list with label 'validation'
    plt.plot(list2, label='validation', marker='x', linestyle='--')
    
    # Adding titles and labels
    plt.title('Training vs. Validation Curves on 10 epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0)  # Setting the y-axis to start at 0
    plt.ylim(top=1)
    plt.legend()
    
    # Display the plot
    plt.grid(True)
    plt.show()

# Main function to execute the program
def main():
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
    plot_barchart(data, m_labels, n_labels)

# Execute the main function
if __name__ == '__main__':
    main()
