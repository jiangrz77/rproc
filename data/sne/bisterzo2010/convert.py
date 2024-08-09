import numpy as np
from pathlib import Path
from util import read_refdata, isotope
# for path in Path('.').iterdir():
# path = Path('L1.25.dat')
# if path.suffix != '.dat':
#     continue
path = Path('/home/jiangrz/ssd/GitHub/rproc/data/sne/bisterzo2010/30ST1.dat')
with open(path, 'r') as file:
    spltlines = file.read().splitlines()
mfrac, nfrac = read_refdata('/home/jiangrz/ssd/GitHub/rproc/data/ref/sol_asplund09.dat')
XH_sol = np.log10(nfrac)
LOGEPS_sol = XH_sol - XH_sol[0] + 12
with open(path, 'w') as file:
    feh = -float(path.stem[:2])/10
    file.write('%-5s%.6E\n'%('H', 1))
    for line in spltlines:
        if line.startswith('#'):
            continue
        name, xfe = line.split()
        xfe = float(xfe)
        iso = isotope(name)
        logeps_sol = LOGEPS_sol[iso.Z-1]
        xh = xfe + feh
        nfrac = np.power(10, xh+logeps_sol-12)
        new_line = '%-5s%.6E\n'%(name, nfrac)
        file.write(new_line)
        # break