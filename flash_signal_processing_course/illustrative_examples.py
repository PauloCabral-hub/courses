# This is a script to test filter capabilities
import function_set as fset
import numpy as np
import matplotlib.pyplot as plt

# simulated signal
fs = 250
#t = np.array(range(10*fs))
t = np.array(range(1*fs))
fake_freq1 = 10  # frequency of the first signal in Hz
fake_freq2 = 20  # frequency of the second signal in Hz
fake_sig1 = 10*np.sin(2 * np.pi * fake_freq1 * (1 / fs) * t)
fake_sig2 = np.sin(2 * np.pi * fake_freq2 * (1 / fs) * t)
fake_sig = fake_sig1 + fake_sig2
fig1, ax1 = plt.subplots(3)
ax1[0].plot(t, fake_sig1, label='10 hz')
# ax1[1].plot(t, fake_sig2, label='20 hz')
# ax1[2].plot(t, fake_sig, label='summed signals')
ax1[0].legend(loc='upper right')
# ax1[1].legend(loc='upper right')
# ax1[2].legend(loc='upper right')
plt.show()

# corresponding spectrum
p_sig, f_hz = fset.fft_spec(fake_sig, fs)
plt.figure()
fset.plot_spec(p_sig, f_hz, 1, 1)

# filtered signal and filter frequency response
fake_filt = fset.filt_func(fs, 15, 5, fake_sig, 4, 1)
# fig2, ax2 = plt.subplots(2)
# ax2[0].plot(t, fake_sig, label='before filtering')
# ax2[1].plot(t, fake_filt, label='after filtering')
# ax2[0].legend(loc='upper right')
# ax2[1].legend(loc='upper right')

# spectrum after filtering
p_sig_filt, f_filt_hz = fset.fft_spec(fake_filt, fs)
plt.figure()
fset.plot_spec(p_sig_filt, f_filt_hz, 1, 1)
