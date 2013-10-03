#/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import glob
from itertools import islice
from scipy.stats import nanmean, nanstd
from rpy import r

def charge(file):
    dat = np.genfromtxt(file, names=True, dtype=None)
    
    return dat


dat = charge('pigments.csv')

dicts = []
nd = {}
for k in dat.dtype.names[4:]: # 'Tratamento','Local','Tempo','Replica'
    nd[k] = []
    nd['name'] = k
    nd['local'] = dat['Local'][0]
    nd['ct0'] = dat[k][(dat['Tratamento']=='INICIO')]
    nd['ct1'] = dat[k][(dat['Tratamento']=='Controle') & (dat['Tempo']=='T1')]
    nd['ct2'] = dat[k][(dat['Tratamento']=='Controle') & (dat['Tempo']=='T2')]
    nd['ft0'] = dat[k][(dat['Tratamento']=='INICIO')]
    nd['ft1'] = dat[k][(dat['Tratamento']=='Fe') & (dat['Tempo']=='T1')]
    nd['ft2'] = dat[k][(dat['Tratamento']=='Fe') & (dat['Tempo']=='T2')]
    nd['dt0'] = dat[k][(dat['Tratamento']=='INICIO')]
    nd['dt1'] = dat[k][(dat['Tratamento']=='DFA') & (dat['Tempo']=='T1')]
    nd['dt2'] = dat[k][(dat['Tratamento']=='DFA') & (dat['Tempo']=='T2')]
    nd['xcontrol'] = np.append(nanmean(nd['ct0']),(nanmean(nd['ct1']), nanmean(nd['ct2'])))
    nd['xferro'] = np.append(nanmean(nd['ft0']),(nanmean(nd['ft1']), nanmean(nd['ft2'])))
    nd['xdfa'] = np.append(nanmean(nd['dt0']),(nanmean(nd['dt1']), nanmean(nd['dt2'])))
    nd['econtrol'] = np.append(nanstd(nd['ct0']),(nanstd(nd['ct1']), nanstd(nd['ct2'])))
    nd['eferro'] = np.append(nanstd(nd['ft0']),(nanstd(nd['ft1']), nanstd(nd['ft2'])))
    nd['edfa'] = np.append(nanstd(nd['dt0']),(nanstd(nd['dt1']), nanstd(nd['dt2'])))
    if nd:
        dicts.append(nd)
    nd = {}


# Figure

for pic in dicts:

    ind = np.array([0, 2, 4])  # x locations for the groups
    width = 0.35       # bars width
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    margin = 0.05
    width = (1.-2.*margin)/3
    
    rects1 = ax.bar((ind+margin), pic['xcontrol'] , width=width,
        facecolor='#000000',yerr=pic['econtrol'] , ecolor='black')
    rects2 = ax.bar((ind+margin+width), pic['xferro'], width=width,
        facecolor='#BEBEBE',  yerr=pic['eferro'], ecolor='black')
    rects3 = ax.bar((ind+margin+2*(width)), pic['xdfa'], width=width, facecolor='#777777', yerr=pic['edfa'], ecolor='black')
    
    ax.set_ylabel(pic['name'] + r"$ (mg . m{^3})$")
    ax.set_title(pic['local']+' - '+'Pigment ' + pic['name'])
    ax.set_xticks(ind+(1.7*width))
    ax.set_xticklabels( ('0', '3', '6') )
    ax.set_xlabel('days')
    ax.legend( (rects1[0], rects2[0], rects3[0]), ('control', '+Fe', '+DFA'), loc='best' )

    plt.savefig(pic['name']+pic['local']+'.png')
    #plt.show()




