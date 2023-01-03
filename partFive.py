import numpy as np
import os, sys, librosa
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import IPython.display as ipd
import pandas as pd
from numba import jit

import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp
from libfmp.b import FloatingBox

from numpy import typing as npt
import typing


def plot_self_similarity(y_ref: npt.ArrayLike, sr: int, affinity: bool = False, hop_length: int = 1024) -> None:
  '''
  To visualize the similarity matrix of the signal

  y_ref: reference signal
  y_comp: signal to be compared
  sr: sampling rate
  affinity: to use affinity or not
  hop_size
  '''


  # Pre-processing stage
  chroma = librosa.feature.chroma_cqt(y=y_ref, sr=sr, hop_length=hop_length)
  chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)


  if not affinity  :
    R = librosa.segment.recurrence_matrix(chroma_stack, k=5)
    imgsim = librosa.display.specshow(R, x_axis='s', y_axis='s',
                                      hop_length=hop_length)
    plt.title('Binary recurrence (symmetric)')
    plt.colorbar()
  
  else :
    R_aff = librosa.segment.recurrence_matrix(chroma_stack, metric='cosine',mode='affinity')
    imgaff = librosa.display.specshow(R_aff, x_axis='s', y_axis='s',
                                      cmap='magma_r', hop_length=hop_length)
    plt.title('Affinity recurrence')
    plt.colorbar()