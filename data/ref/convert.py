from pathlib import Path
path = 'sol_asplund09.dat'
with open(path, 'r') as file:
    spltlines = file.read().splitlines()
with open(path, 'w') as file:
    for line in spltlines:
        if line.startswith('#'):
            continue
        # symbol, Z, ATW, logn, nratio_H, nratio_He, mfrac  = line.split()
        symbol, nratio_H, mfrac  = line.split()
        symbol = '%-5s'%symbol
        new_line = '   '.join([symbol, mfrac, nratio_H, '\n'])
        file.write(new_line)