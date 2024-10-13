#!/usr/env python3
import main_ilp
import time
import os
import sys

def eval_one(m, p, n, taskID):
  start = time.time_ns()
  main_ilp.outsideCall('dag_m%ip%in%i_input_files/multicore_system_DAGtask_%i.json' % (m,p,n, taskID), '/tmp/dag_output_m%ip%in%i_id%i' % (m,p,n,taskID))
  end = time.time_ns()
  with open('time_results_m%ip%in%i' % (m,p,n), 'a') as res_file:
    res_file.write('%f,' % ((end - start) * 1e-3)) #in µs


def get_list_ids(m,p,n):
  listIDs = [ int(filename.split('_')[2].split('.')[0]) for filename in os.listdir('dag_m%ip%in%i_output_schedules' % (m,p,n))]
  if len(listIDs) > 100:
    listIDs = listIDs[:100]
  return listIDs



m=int(sys.argv[1])
p=int(sys.argv[2])
n=int(sys.argv[3])
for taskID in get_list_ids(m,p,n):
  eval_one(m,p,n,taskID)