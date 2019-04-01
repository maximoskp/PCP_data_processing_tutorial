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
# import tensorflow as tf
from sklearn.decomposition import NMF
import io

# MAKE DATA ===================================================================
# the user should give the file name - the folder with the files
folderName = 'BachChorales'
# folderName = 'BachChoralesSmall'

# the user should give the parts to be considered for multi-hot output
parts_for_surface = [0, 1]
parts_for_tonality = [2]

# time resolution should be set by the user
time_res = 16

m,t,full_mat = s2n.get_concat_rel_pcp_np_from_folder(folderName, parts_for_surface, time_res, parts_for_tonality, bin_out=True)

# NNMF
plt.imshow(m[:,0:48], cmap='gray_r', interpolation='none'); plt.savefig('figs/m.png', dpi=500); plt.clf()
model = NMF(n_components = 3, init='random', random_state=0)
w = model.fit_transform(m)
h = model.components_
plt.imshow(w, cmap='gray_r', interpolation='none'); plt.savefig('figs/w.png'); plt.clf()
plt.imshow(h[:,0:48], cmap='gray_r', interpolation='none'); plt.savefig('figs/h.png', dpi=500); plt.clf()
# new estimation
m_ = np.matmul(w,h)
plt.imshow(m_[:,0:48], cmap='gray_r', interpolation='none'); plt.savefig('figs/m_.png', dpi=500); plt.clf()

# write composition
with open('full-file.txt', 'w') as f:
    np.savetxt(f, full_mat[:,200:2000], delimiter='\t', newline='\n', fmt='%.2f')

# write H
with open('h-file.txt', 'w') as f:
    np.savetxt(f, h[:,200:2000], delimiter='\t', newline='\n', fmt='%.2f')