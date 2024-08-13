import re
import numpy as np
import pandas as pd
import sys
sys.path.append('../../..')
from util import isotope, read_refdata

mfrac_sol, nfrac_sol = read_refdata('/home/jiangrz/ssd/GitHub/rproc/data/ref/sol_asplund09.dat')
datapath = '/home/jiangrz/ssd/GitHub/rproc/data/sne/prantzos2020/SolAbu_srp.csv'
df_SolAbu_srp = pd.read_csv(datapath, header=[0])

with open('s_component.dat', 'w') as s_file:
    for _, row in df_SolAbu_srp.iterrows():
        element = isotope(row['Elm.'])
        Z = element.Z
        sfrac = row['s-']
        nfrac = nfrac_sol[Z-1]*sfrac
        new_line = '%-5s%.6E\n'%(element.symbol, nfrac)
        s_file.write(new_line)

with open('r_component.dat', 'w') as r_file:
    for _, row in df_SolAbu_srp.iterrows():
        element = isotope(row['Elm.'])
        Z = element.Z
        rfrac = row['r-']
        nfrac = nfrac_sol[Z-1]*rfrac
        new_line = '%-5s%.6E\n'%(element.symbol, nfrac)
        r_file.write(new_line)

with open('p_component.dat', 'w') as p_file:
    for _, row in df_SolAbu_srp.iterrows():
        element = isotope(row['Elm.'])
        Z = element.Z
        pfrac = row['p-']
        nfrac = nfrac_sol[Z-1]*pfrac
        new_line = '%-5s%.6E\n'%(element.symbol, nfrac)
        p_file.write(new_line)