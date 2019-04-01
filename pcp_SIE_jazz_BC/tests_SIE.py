#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:02:22 2018

@author: maximoskaliakatsos-papakostas
"""

'''
binary columns to decimal
lala = composition.transpose().dot(1 << np.arange(composition.shape[0] - 1, -1, -1))
'''

from music21 import *
import numpy as np
import score2np as s2n
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
import io

# MAKE DATA ===================================================================
# the user should give the file name - the folder with the files
data_1 = 'BachChorales'
data_2 = 'Jazz'

# the user should give the parts to be considered for multi-hot output
parts_for_surface = [0, 1]
parts_for_tonality = [2]

# time resolution should be set by the user
time_res = 16

#m,t,full_mat = s2n.get_concat_rel_pcp_np_from_folder(data_1, parts_for_surface, time_res, parts_for_tonality, bin_out=True)

pcps_1, sies_1 = s2n.get_separate_rel_pcp_and_sie(data_1, parts_for_surface, time_res, parts_for_tonality, bin_out=True)
pcps_2, sies_2 = s2n.get_separate_rel_pcp_and_sie(data_2, parts_for_surface, time_res, parts_for_tonality, bin_out=True)

# concatenate pcps into a single matrix ROWS: pieces, 12 columns
concat_pcps_1 = np.array(pcps_1)
concat_pcps_2 = np.array(pcps_2)

pca = PCA(n_components=2)
pca_1 = pca.fit_transform(concat_pcps_1)
pca_2 = pca.fit_transform(concat_pcps_2)

plt.plot(pca_1[:,0], pca_1[:,1], 'x')
for i in range(pca_1.shape[0]):
    plt.text( pca_1[i,0], pca_1[i,1], '1-'+str(i) )
plt.plot(pca_2[:,0], pca_2[:,1], 'o')
for i in range(pca_2.shape[0]):
    plt.text( pca_2[i,0], pca_2[i,1], '2-'+str(i) )
plt.legend()
plt.savefig('figs/pca_pcps.png', dpi=500); plt.clf()

for i, pcp in enumerate(pcps_1):
    plt.bar(range(len(pcp)), pcp);plt.savefig('figs/1-'+str(i)+'.png', dpi=500); plt.clf()
for i, pcp in enumerate(pcps_2):
    plt.bar(range(len(pcp)), pcp);plt.savefig('figs/2-'+str(i)+'.png', dpi=500); plt.clf()

plt.boxplot( [np.array(sies_1), np.array(sies_2)] );plt.savefig('figs/boxplot.png', dpi=500); plt.clf()

'''
# NNMF
plt.imshow(m[:,100:148], cmap='gray_r', interpolation='none'); plt.savefig('figs/m.png'); plt.clf()
model = NMF(n_components = 3, init='random', random_state=0)
w = model.fit_transform(m)
h = model.components_
plt.imshow(w, cmap='gray_r', interpolation='none'); plt.savefig('figs/w.png'); plt.clf()
plt.imshow(h[:,100:148], cmap='gray_r', interpolation='none'); plt.savefig('figs/h.png'); plt.clf()
# new estimation
m_ = np.matmul(w,h)
plt.imshow(m_[:,100:148], cmap='gray_r', interpolation='none'); plt.savefig('figs/m_.png'); plt.clf()

# write composition
with open('full-file.txt', 'w') as f:
    np.savetxt(f, full_mat[:,200:2000], delimiter='\t', newline='\n', fmt='%.2f')

# write H
with open('h-file.txt', 'w') as f:
    np.savetxt(f, h[:,200:2000], delimiter='\t', newline='\n', fmt='%.2f')
'''