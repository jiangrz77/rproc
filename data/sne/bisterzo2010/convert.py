import numpy as np
import pandas as pd
from pathlib import Path
from util import read_refdata, isotope
# path = Path('L1.25.dat')
# if path.suffix != '.dat':
#     continue
path_list = [
    Path('/home/jiangrz/ssd/GitHub/rproc/data/sne/bisterzo2010/AGB_m13.csv'), 
    Path('/home/jiangrz/ssd/GitHub/rproc/data/sne/bisterzo2010/AGB_m15.csv')
]

mfrac_sol, nfrac_sol = read_refdata(
    '/home/jiangrz/ssd/GitHub/rproc/data/ref/sol_asplund09.dat')
XH_sol = np.log10(
    nfrac_sol, 
    where=nfrac_sol>0, 
    out=np.full(len(nfrac_sol), np.nan))
LOGEPS_sol = XH_sol - XH_sol[0] + 12
for rpath in Path('/home/jiangrz/ssd/GitHub/rproc/data/sne/bisterzo2010').iterdir():
    if rpath.suffix != '.csv':
        continue
    # with open(rpath, 'r') as file:
    #     spltlines = file.read().splitlines()
    r_df = pd.read_csv(rpath)
    if 'ST' in rpath.stem:
        st = int(rpath.stem[6:-3])/10
        m = int(rpath.stem[-2:])/10
        init_feh = r_df.columns[1:]
        for feh in init_feh:
            wpath = rpath.parent/('%02dST%04dm%02d.dat'%(np.abs(float(feh)*10), st*10, m*10))
            # with open(wpath, 'w') as file:
            El = r_df.loc[:, 'El'].values
            XFe = r_df.loc[:, feh].values.astype(float)
            with open(wpath, 'w') as wfile:
                wfile.write('%-5s%.6E\n'%('H', 1.0))
                for el, xfe in zip(El, XFe):
                    iso = isotope(el)
                    logeps_sol = LOGEPS_sol[iso.Z-1]
                    xh = xfe + float(feh)
                    nfrac = np.power(10, xh+logeps_sol-12)
                    new_line = '%-5s%.6E\n'%(iso.symbol, nfrac)
                    wfile.write(new_line)
    elif 'FeH' in rpath.stem:
        init_st = r_df.columns[1:]
        m = int(rpath.stem[-2:])/10
        feh = -int(rpath.stem[7:-3])/10
        if feh > -2.4:
            continue
        for st in init_st:
            wpath = rpath.parent/('%02dST%04dm%02d.dat'%(np.abs(feh*10), float(st)*10, m*10))
            # with open(wpath, 'w') as file:
            El = r_df.loc[:, 'El'].values
            XFe = r_df.loc[:, st].values.astype(float)
            with open(wpath, 'w') as wfile:
                wfile.write('%-5s%.6E\n'%('H', 1.0))
                for el, xfe in zip(El, XFe):
                    iso = isotope(el)
                    logeps_sol = LOGEPS_sol[iso.Z-1]
                    xh = xfe + feh
                    nfrac = np.power(10, xh+logeps_sol-12)
                    new_line = '%-5s%.6E\n'%(iso.symbol, nfrac)
                    wfile.write(new_line)

        #     for line in spltlines:
        #         if line.startswith('#'):
        #             continue
        #         name, xfe = line.split()
        #         xfe = float(xfe)
        #         iso = isotope(name)
        #         logeps_sol = LOGEPS_sol[iso.Z-1]
        #         xh = xfe + feh
        #         nfrac = np.power(10, xh+logeps_sol-12)
        #         new_line = '%-5s%.6E\n'%(name, nfrac)
        #         file.write(new_line)
        #         # break