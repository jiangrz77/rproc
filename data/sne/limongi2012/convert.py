import os
import numpy as np
import pandas as pd
from matplotlib import colormaps as cmap
from starfit import DB
from starfit.autils.stardb import StarDB
from starfit.autils.abusets import SolAbu
from starfit.utils import find_data

import matplotlib.pyplot as plt
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

db = 'znuc_lc12.el.y.stardb.xz'
dbpath = find_data(DB, db)
db = StarDB(dbpath, silent=True)
sun = SolAbu('solas09.dat', silent=True)
raw_fielddata = db.fielddata
raw_fieldname = db.fieldnames#[db.fieldflags==0]
raw_model_abund = db.data
raw_element = [_ion.name().capitalize() for _ion in db.ions]
element_list_abvC = symbol_list
model_logeps_df = pd.DataFrame(columns=element_list_abvC)
fielddata_df = pd.DataFrame(columns=raw_fieldname)
for _abu, _el in zip(raw_model_abund.T, raw_element):
    if _el in element_list_abvC:
        model_logeps_df[_el] = np.log10(
            _abu, where=_abu>0, 
            out=np.full(len(_abu), np.nan))
for _fieldname in raw_fieldname:
    fielddata_df[_fieldname] = raw_fielddata[_fieldname]
maxns = 2.0
# fielddata_df = fielddata_df.loc[fielddata_df.loc[:, 'remnant']<=maxns, :]
M_ccsn = np.array(model_logeps_df.values).astype(float)
P_ccsn = np.array(fielddata_df.values[:, :]).astype(float)
P_ccsn[:, 2] = np.log10(P_ccsn[:, 2], where=P_ccsn[:, 2]>0, out=-3.2*np.ones(P_ccsn.shape[0]))

for _P, _M in zip(P_ccsn, raw_model_abund):
    string_property = '%04d%03d%03d%06d'%(_P[0]*10, _P[1]*10, _P[2]*10, _P[3]*1000)
    with open('/home/jiangrz/ssd/GitHub/rproc/data/sne/hw2010/%s.dat'%string_property, 'w') as file:
        for _symbol, _yield in zip(symbol_list, _M):
            file.write('%-5s%.8E\n'%(_symbol, _yield))