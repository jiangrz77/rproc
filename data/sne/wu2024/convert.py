import re
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/jiangrz/ssd/GitHub/rproc/')
from util import isotope, read_refdata

mfrac_sol, nfrac_sol = read_refdata('/home/jiangrz/ssd/GitHub/rproc/data/ref/sol_asplund09.dat')
datapath = '/home/jiangrz/ssd/GitHub/rproc/data/sne/wu2024/collapsar_average_yz.txt'
wrtiepath = '/home/jiangrz/ssd/GitHub/rproc/data/sne/wu2024/collapsar.dat'

with open(datapath, 'r') as rfile:
    spltlines = rfile.read().splitlines()
    
with open(wrtiepath, 'w') as wfile:
    for line in spltlines[1:]:
        Z, Yi = line.split()
        Z = int(Z)
        if Z < 1:
            continue
        try:
            El = isotope(Z)
            Yi = float(Yi)
            new_line = '%-5s%.6E\n'%(El.symbol, Yi)
            wfile.write(new_line)
        except ValueError:
            continue