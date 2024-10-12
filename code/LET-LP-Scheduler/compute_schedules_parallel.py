#!/usr/bin/env python3
import multiprocessing as multip
import os

def getMissingFilesList():
    p=8
    n_list=[10,20,30,40,50]
    m_list=[2,4,6,7,8]
    missing_files_list = []

    for n in n_list:
        for m in m_list:
          if not (m == 2 and (n == 30 or n == 40 or n == 50)):
            folder = "dag_m%ip%in%i_input_files" % (m,p,n)
            folder_output = "dag_m%ip%in%i_output_schedules" % (m,p,n)
            with open("missing_file_ids_m%ip%in%i" % (m, p, n), 'r') as id_file:
                for line in id_file:
                    id = line.strip()
                    filename = "%s/multicore_system_DAGtask_%s.json" % (folder, id)
                    missing_files_list.append({"input_name": filename, "output_name": "%s/schedule_dag_%s.json" % (folder_output, id)})

    return missing_files_list


def compute_schedules(start_id, end_id, files_list):

    for id in range(start_id, end_id):
        input_file_name = files_list[id]['input_name']
        output_file_name = files_list[id]['output_name']
        os.system("python3 main_ilp.py --infile \"%s\" --outfile \"%s\" --solver GUROBI > /tmp/%s.log" % (input_file_name, output_file_name, id))


    return 0

def main():

    missingFilesList = getMissingFilesList()
    numProcs = 90
    numTasks = len(missingFilesList)

    task_step = numTasks // numProcs
    processes = []

    for incr in range(numProcs):

        start_task_id = task_step * incr
        end_task_id = task_step * (incr+1)
        process = multip.Process(target=compute_schedules, args=(start_task_id, end_task_id, missingFilesList))
        processes.append(process)
        process.start()

    

    for process in processes:
        process.join()

    print("finished computing")


    


if __name__ == '__main__':

    main()