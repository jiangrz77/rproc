from pathlib import Path
# for path in Path('.').iterdir():
path = Path('L1.25.dat')
# if path.suffix != '.dat':
#     continue
with open(path, 'r') as file:
    spltlines = file.read().splitlines()
with open(path, 'w') as file:
    for line in spltlines:
        if line.startswith('#'):
            continue
        name, Z, N, A, X, Y = line.split()
        name = '%-5s'%name
        new_line = '   '.join([name, X, Y, '\n'])
        file.write(new_line)