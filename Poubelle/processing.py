#/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import nanmean, nanstd

def extract_pigments(indir):
    '''
    extract pigments data of *.csv files, and return a list of
    dictionaries.

    INPUT
    -----
    indir : str 
    string of specific directory

    OUTPUT
    ------
    var: list of dictionaries, containing pigments and informations.

    '''
    dicts = []
    for filename in sorted(glob.glob(os.path.join(indir, '*csv'))):
        dat = np.genfromtxt(filename, names=True, dtype=None)
        
        #dicts = []
        nd = {}
        parse = ['Tratamento','Local','Tempo','Replica']
        
        for k in dat.dtype.names:
            if k in parse:
                continue
                
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

    return dicts


def explot(var):
    '''
    Export Plots of pigments in respective folder.
    
    INPUT
    -----
    var : List of dictionaries
    List of dictionaries containing pigments information.

    OUTPUT
    ------
    Save plots as *.png, in specific folder, named as sited choice.
    '''
    for pic in var:
    
        ind = np.array([0, 2, 4])  # x locations for the groups
        width = 0.35       # bars width
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        margin = 0.05
        width = (1.-2.*margin)/3
        
        rects1 = ax.bar((ind+margin), pic['xcontrol'] , width=width,\
            facecolor='#000000',yerr=pic['econtrol'] , ecolor='black')
        rects2 = ax.bar((ind+margin+width), pic['xferro'], width=width,\
            facecolor='#BEBEBE',  yerr=pic['eferro'], ecolor='black')
        rects3 = ax.bar((ind+margin+2*(width)), pic['xdfa'],\
            width=width,   facecolor='#777777', yerr=pic['edfa'],\
            ecolor='black')
        
        ax.set_ylabel(pic['name'] + r"$ (\mu g . m{^3})$")
        ax.set_title(pic['local']+' - '+'Pigment ' + pic['name'])
        ax.set_xticks(ind+(1.7*width))
        ax.set_xticklabels( ('0', '3', '6') )
        ax.set_xlabel('days')
        ax.legend( (rects1[0], rects2[0], rects3[0]), ('control',\
        '+Fe', '+DFA'), loc='best' )

        figname = (pic['name']+pic['local']+'.png')
        outdir = ('figs_'+pic['local'])
        
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        plt.savefig(os.path.join(outdir,figname))
        plt.clf()


def explot_chla(var):
    '''
    Export Plots of pigments padronized by chlorophyll-a.
    
    INPUT
    -----
    var : List of dictionaries, containing pigments information.

    OUTPUT
    ------
    Save plots as *.png, in specific folder.
    '''

    ch_c0 = var[0]['ct0']
    ch_c1 = var[0]['ct1']
    ch_c2 = var[0]['ct2']
    ch_f0 = var[0]['ft0']
    ch_f1 = var[0]['ft1']
    ch_f2 = var[0]['ft2']
    ch_d0 = var[0]['dt0']
    ch_d1 = var[0]['dt1']
    ch_d2 = var[0]['dt2']
    
    for pic in var:
        xpic_c = [nanmean(pic['ct0']/ch_c0), nanmean(pic['ct1']/ch_c1), nanmean(pic['ct2']/ch_c2)]

        epic_c = [nanstd(pic['ct0']/ch_c0), nanstd(pic['ct1']/ch_c1), nanstd(pic['ct2']/ch_c2)]

        xpic_f = [nanmean(pic['ft0']/ch_f0), nanmean(pic['ft1']/ch_f1), nanmean(pic['ft2']/ch_f2)]

        epic_f = [nanstd(pic['ft0']/ch_f0), nanstd(pic['ft1']/ch_f1), nanstd(pic['ft2']/ch_f2)]

        xpic_d = [nanmean(pic['dt0']/ch_d0), nanmean(pic['dt1']/ch_d1), nanmean(pic['dt2']/ch_d2)]
        
        epic_d = [nanstd(pic['dt0']/ch_d0), nanstd(pic['dt1']/ch_d1), nanstd(pic['dt2']/ch_d2)]

        ind = np.array([0, 2, 4])  # x locations for the groups
        width = 0.35       # bars width
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        margin = 0.05
        width = (1.-2.*margin)/3
        
        rects1 = ax.bar((ind+margin), xpic_c , width=width, facecolor='#000000',yerr=epic_c , ecolor='black')

        rects2 = ax.bar((ind+margin+width), xpic_f, width=width,  facecolor='#BEBEBE',  yerr=epic_f, ecolor='black')

        rects3 = ax.bar((ind+margin+2*(width)), xpic_d, width=width,   facecolor='#777777', yerr=epic_d, ecolor='black')
        
        ax.set_ylabel(pic['name'] + r"$ (\mu g . m{^3})$")
        ax.set_title(pic['local']+' - '+ pic['name'] +'/Chl-a')
        ax.set_xticks(ind+(1.7*width))
        ax.set_xticklabels( ('0', '3', '6') )
        ax.set_xlabel('days')
        ax.legend( (rects1[0], rects2[0], rects3[0]), ('control',\
        '+Fe', '+DFA'), loc='best' )

        figname = (pic['name']+pic['local']+'Chla'+'.png')
        outdir = ('figs_chla'+pic['local'])
        
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        plt.savefig(os.path.join(outdir,figname))
        plt.clf()
