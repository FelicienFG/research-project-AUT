import matplotlib.pyplot as plt

# Function to read data from file and return two lists of values
def read_data_from_file(filename):
    with open(filename, 'r') as file:
        # Reading lines from the file
        lines = file.readlines()

        # Assuming the file has exactly two lines of comma-separated values
        list1 = list(map(float, lines[0].strip().split(',')))
        list2 = list(map(float, lines[1].strip().split(',')))

    return list1, list2

# Function to plot two lists as curves
def plot_curves(list1, list2):
    plt.figure(figsize=(10, 5))
    
    # Plotting the first list with label 'train'
    plt.plot(list1, label='train', marker='o', linestyle='-')
    
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
    filename = 'results_lossaccu'  # Change this to your file's path
    list1, list2 = read_data_from_file(filename)
    plot_curves(list1, list2)

# Execute the main function
if __name__ == '__main__':
    main()
