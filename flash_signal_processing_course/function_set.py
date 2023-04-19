# This file contains a set of functions to use for the eeg data analysis
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
from scipy import signal
import warnings


def scope_parameters(sig):
    scope = max(sig) - min(sig)
    y_min = min(sig) - 1 / 10 * scope
    y_max = max(sig) + 1 / 10 * scope
    return y_min, y_max


def plot_time(time, sig, x_max, y_min, y_max, channel, show):
    # title_str = 'Signal in Channel ' + str(channel)
    plt.plot(time, sig)
    if show == 1:
        plt.axis([0, x_max, y_min, y_max])
        plt.xlabel('time (sec.)', labelpad=1)
        plt.ylabel('uV', labelpad=1)
    return


def fft_spec(sig, fs):
    aux = sig.shape[0] % 2
    if aux != 0:
        sig = np.append(sig, [0])
    f_sig = (1 / sig.shape[0]) * np.fft.fft(sig)  # apply the normalization factor
    p_sig = np.absolute(f_sig)[0: int(sig.shape[0] / 2 + 1)]
    l_f = p_sig.shape[0]
    p_sig[1:] = 2 * p_sig[1:]
    f_hz = (fs / sig.shape[0]) * np.array(range(l_f))
    return p_sig, f_hz


def plot_spec(p_sig, f_hz, channel, show):
    # title_str2 = 'Spectrum of Channel ' + str(channel)
    plt.stem(f_hz, p_sig)
    if show == 1:
        plt.xlabel('freq (Hz)', labelpad=1)
        plt.ylabel('|P(f)| (microV/Hz)', labelpad=1)
        plt.xlim(0, max(f_hz))
    return


def segment_data(sig, n_win):
    win_len = int(sig.shape[0] / n_win)
    b = 0
    e = -1
    sig_epoc_list = np.zeros((win_len, n_win))
    for i in range(30):
        b = e + 1
        e = b + win_len - 1
        sig_epoc_list[:, i] = sig[b:e + 1]
    return sig_epoc_list


def rect_sig(sig, fs, r_factor):
    flag_vec = np.zeros(sig.shape[0], )
    flag = r_factor * np.std(sig)
    for i in range(sig.shape[0]):
        if abs(sig[i]) > flag:
            flag_vec[i] = float(1)

    aux_total = flag_vec - 0.5
    flag_pos = np.where(aux_total > 0)[0]
    middle = int(sig.shape[0] / 2)
    left_cut = np.max(np.where(flag_pos < middle))
    right_cut = np.min(np.where(flag_pos > middle))
    sig = sig[flag_pos[left_cut]:flag_pos[right_cut]]
    time = (1 / fs) * np.arange(sig.shape[0])
    return time, sig


def filt_func(fs, l_fc, h_fc, sig, order, show):
    if l_fc != 0:
        wl = l_fc / (fs / 2)  # normalizing frequency to pass to the design
        bl, al = signal.butter(order, wl, 'low')  # getting the filter coefficients
        sig_filt = signal.filtfilt(bl, al, sig)  # filtering the signal
        sig = sig_filt
    if h_fc != 0:
        wh = h_fc / (fs / 2)  # normalizing frequency to pass to the design
        bh, ah = signal.butter(order, wh, 'high')  # getting the filter coefficients
        sig_filt = signal.filtfilt(bh, ah, sig)  # filtering the signal
        sig = sig_filt
    if show == 1:
        fig, ax = plt.subplots()
        ax.set_title('Digital filter frequency response')
        ax.set_ylabel('Amplitude [dB]', color='b')
        ax.set_xlabel('Frequency [Hz]')
        if l_fc != 0:
            wl, hl = signal.freqz(bl, al)
            fl = wl * (fs / (2 * np.pi))  # since w is the normalized frequency from [0,pi)
            ax.plot(fl, 20 * np.log10(abs(hl)), 'b')
            plt.axvline(l_fc, color='red')  # cutoff frequency
        if h_fc != 0:
            wh, hh = signal.freqz(bh, ah)
            fh = wh * (fs / (2 * np.pi))  # since w is the normalized frequency from [0,pi)
            ax.plot(fh, 20 * np.log10(abs(hh)), 'b')
            plt.axvline(h_fc, color='red')  # cutoff frequency
        ax.set_xlim(0, fs / 2)
        plt.grid()
    return sig_filt


def sn_ratio(p_sig, f_hz, x_inf, y_inf):
    warnings.filterwarnings("ignore")
    f = sci.interpolate.interp1d(f_hz, p_sig)
    sig_int, _ = sci.integrate.quad(f, x_inf, y_inf)
    total_int, _ = sci.integrate.quad(f, f_hz[0], f_hz[-1])
    noise_int = total_int - sig_int
    snr = sig_int / noise_int  # higher values are better
    return snr


def y_lim_for_plt(data_matrix):
    y_lim = np.max(abs(data_matrix))
    return y_lim
