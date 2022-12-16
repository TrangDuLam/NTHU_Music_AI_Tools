import librosa
from librosa import display
from librosa import feature

import numpy as np
from numpy import typing as npt

from matplotlib import pyplot as plt


def show_duration(y: npt.ArrayLike, sr: int) -> float:
    pass


def selcet_time(start_time: float, end_time: float) :
    pass


def plot_waveform(y: npt.ArrayLike, sr: int, start_time: float = 0.0, end_time: float = None) -> None :

    startIdx = int(start_time * sr)
    
    if not end_time :
        librosa.display.waveshow(y[startIdx:], sr)
    
    else :
        endIdx = int(end_time * sr)
        librosa.display.waveshow(y[startIdx:endIdx - 1], sr)   
    
    return


def signal_RMS_analysis(y: npt.ArrayLike, show_plot: bool = True, to_csv: bool = False) :

    rms = librosa.feature.rms(y = y)
    times = librosa.times_like(rms)

    if show_plot :
        plt.plot(times, rms[0])

    if to_csv :
        info = np.vstack((times, rms[0])).T
        np.savetxt('time_to_rms.csv', info, fmt="%.3f", delimiter=",")