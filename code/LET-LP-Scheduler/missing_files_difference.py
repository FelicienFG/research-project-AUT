import sys


files = list(sys.argv[1:])
id_sets = []

for file in files:

    with open(file, 'r') as infile:
        id_set = set(int(line.strip()) for line in infile)

        id_sets.append({'file': file, 'ids': id_set})


for i in range(len(id_sets) - 1):
    for j in range(i+1, len(id_sets)):
        print('file ', id_sets[j]['file'], ' - ', id_sets[i]['file'], ' = \n')
        print(id_sets[j]['ids'] - id_sets[i]['ids'])


    
    