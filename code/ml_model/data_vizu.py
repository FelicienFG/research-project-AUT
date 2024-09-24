import matplotlib.pyplot as plt
import pandas as pd

# Function to read data from file and return two lists of values
def read_data_from_file(filename):
    with open(filename, 'r') as file:
        # Reading lines from the file
        lines = file.readlines()

        # Assuming the file has exactly two lines of comma-separated values
        list1 = list(map(float, lines[2].strip().split(',')))
        list2 = list(map(float, lines[3].strip().split(',')))

    return list1, list2

def plot_barchart(csv_file):

    # Read the CSV file
    df = pd.read_csv(csv_file)
    print(df)
    # Assuming the first two columns are the ones to be used for the bars
    first_column = df.columns[0]
    second_column = df.columns[1]

    # Create the bar chart
    plt.bar([first_column, second_column], [df[first_column].mean(), df[second_column].mean()], width=0.4, label=[first_column, second_column], color=['blue', 'red'], align='center')
    #plt.bar([first_column, second_column], df[second_column].mean(), width=0.4, label=second_column, align='center')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Makespan')
    plt.title('Average makespan for the Model and ILP methods')

    # Show legend
    plt.legend()

    # Display the plot
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
    filename = 'results_makespan'  # Change this to your file's path
    #list1, list2 = read_data_from_file(filename)
    #plot_curves(list1, list2)
    plot_barchart(filename)

# Execute the main function
if __name__ == '__main__':
    main()
