#!/usr/bin/env python3
import numpy as np


with open("progress.progress", 'r') as progress_file:
  percentages = []
  for line in progress_file:
    outfiles, infiles = line.split('/')
    outfiles, infiles = int(outfiles), int(infiles)
    
    percentages.append(float(outfiles) / infiles)

  print("total percentage: %f%%" % (np.mean(np.array(percentages)) * 100.0))