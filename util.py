import socket
import re, scipy
import numpy as np
from pathlib import Path
log10 = lambda X: np.log10(X, 
    out=np.full(X.shape, np.nan), 
    where=X>0)

symbol_arr = np.array([
    'h',  'he', 'li', 'be', 'b',  'c',  'n',   'o',  'f', 'ne', 
    'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',  'k', 'ca',
    'sc', 'ti', 'v',  'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn',
    'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr',  'y', 'zr',
    'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 
    'sb', 'te', 'i',  'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 
    'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 
    'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt', 'au', 'hg', 
    'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 
    'pa', 'u'
])
symbol_arr = np.char.capitalize(symbol_arr)
symbol_list = symbol_arr.tolist()
def Symbol2Z(Symbol):
    if isinstance(Symbol, str):
        Z = symbol_list.index(Symbol)+1
    elif hasattr(Symbol, '__iter__'):
        Z = np.array([symbol_list.index(_) for _ in Symbol])+1
    return Z
def Z2Symbol(Z):
    Symbol = symbol_arr[Z-1]
    return Symbol

class isotope():
    def __init__(self, identifier, type=None) -> None:
        if isinstance(identifier, str):
            string_pattern = r'([A-Za-z]{1,2})([0-9]*)'
            re_result = re.match(string_pattern, identifier)
            symbol, A = re_result.groups()
            symbol = symbol.lower().capitalize()
            if symbol in symbol_arr:
                Z = Symbol2Z(symbol)
                if A == '':
                    A = None
                else:
                    A = int(A)
                    if A < Z:
                        raise ValueError('Mass Number should be larger than Atomic Number.')
            else:
                raise ValueError('Symbol cannot be identified. Note that element beyond U is not allowed!')
        elif isinstance(identifier, int):
            Z = identifier
            if Z <= 92:
                symbol = Z2Symbol(Z)
                self.symbol = symbol
                A = None
            else:
                raise ValueError('Element beyond U is not allowed!')
        self.symbol = symbol
        self.Z = Z
        self.A = A
        if type is not None:
            self.type = type
        else:
            if self.A is None:
                self.type = 'Element'
            else:
                self.type = 'Isotope'
        if self.type == 'Element':
            self.identifier = r'Element %s'%(self.symbol)
        elif self.type == 'Isotope':
            self.identifier = r'Isotope %s%d'%(self.symbol, self.A)
    
    def __repr__(self):
        return self.identifier
    
    def __del__(self):
        pass

def read_refdata(filename):
    mfrac_arr = np.zeros(len(symbol_arr), dtype=np.float64)
    nfrac_arr = np.zeros(len(symbol_arr), dtype=np.float64)
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError('%s not found. '%filepath)
    with open(filepath, 'r') as rfile:
        lines = rfile.read().splitlines()
    iso_list = []
    mfrac_list = []
    nfrac_list = []
    for line in lines:
        linesplt = line.split()
        if len(linesplt) == 3:
            iso, massfrac, numfrac = line.split()
            iso = isotope(iso)
        elif len(linesplt) == 2:
            iso, massfrac = line.split()
            iso = isotope(iso)
            if iso.A is None:
                numfrac = massfrac
                massfrac = -1.00
        mfrac = float(massfrac)
        nfrac = float(numfrac)
        iso_list.append(iso)
        mfrac_list.append(mfrac)
        nfrac_list.append(nfrac)
    for iso, mfrac, nfrac in zip(iso_list, mfrac_list, nfrac_list):
        mfrac_arr[iso.Z-1] += mfrac
        nfrac_arr[iso.Z-1] += nfrac
    # mfrac_arr /= np.sum(mfrac_arr)
    # nfrac_arr /= np.sum(nfrac_arr)
    return mfrac_arr, nfrac_arr

def load_yield(snref_path):
    nfrac_list = []
    P_list = []
    str_split = np.loadtxt(snref_path/'p_detail', max_rows=1, dtype=int)
    p_mod = np.loadtxt(snref_path/'p_detail', skiprows=1, dtype=int)
    for path in snref_path.iterdir():
        if path.suffix == '.dat':
            mfrac, nfrac = read_refdata(path)
            nfrac_list.append(nfrac)
            # break
            P_vec = np.array([
                path.stem[str_split[idx]:str_split[idx+1]] 
                for idx in range(len(str_split)-1)], 
                dtype=np.float64)
            P_vec /= p_mod
            P_list.append(P_vec)
    M_nfrac = np.array(nfrac_list)
    M_logeps = log10(M_nfrac)
    M_logeps = M_logeps - M_logeps[:, [0]] + 12 # - M[:, [25]]
    P = np.array(P_list)
    return P, M_logeps

def load_solref(solref_path='../data/ref/sol_asplund09.dat'):
    solref_path = Path(solref_path)
    sol_mfrac, sol_nfrac = read_refdata(solref_path)
    sol_logeps = log10(sol_nfrac)
    sol_logeps += (-sol_logeps[0]+12)
    return sol_logeps


hostname = socket.gethostname()
if hostname == 'jerome-linux':
    hostdir = Path('/home/jerome/Documents/GitHub')
elif hostname == 'sage2020':
    hostdir = Path('/home/jiangrz/hdd23')
