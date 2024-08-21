import re, scipy
import numpy as np
from pathlib import Path

symbol_list = [
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
]
symbol_list = [_.capitalize() for _ in symbol_list]
Symbol2Z_dict = {_:_idx+1 for _idx, _ in enumerate(symbol_list)}
Z2Symbol_dict = {_idx+1:_ for _idx, _ in enumerate(symbol_list)}

class isotope():
    def __init__(self, identifier, type=None) -> None:
        if isinstance(identifier, str):
            string_pattern = r'([A-Za-z]{1,2})([0-9]*)'
            re_result = re.match(string_pattern, identifier)
            symbol, A = re_result.groups()
            symbol = symbol.lower().capitalize()
            if symbol in symbol_list:
                Z = Symbol2Z_dict[symbol]
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
                symbol = Z2Symbol_dict[Z]
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
    mfrac_arr = np.zeros(len(symbol_list), dtype=np.float64)
    nfrac_arr = np.zeros(len(symbol_list), dtype=np.float64)
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
        else:
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