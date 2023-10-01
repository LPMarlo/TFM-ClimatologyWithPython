import math
import numpy as np
import scipy.fftpack as fftpack
from copy import deepcopy


def fourier_transform(signal, spacing):
    sig_fft = fftpack.fft(signal)
    sample_freq = fftpack.fftfreq(len(signal), d=spacing) * len(signal) * spacing
    sample_freq = np.resize(sample_freq, sig_fft.shape)
    return sig_fft, sample_freq


def spectrum(signal_fft, freqs, scaling='amplitude', variance=None):
    assert scaling in ['amplitude', 'power', 'R2']
    if scaling == 'R2':
        assert variance, \
            "To calculate variance explained must provide variance value"

    if len(signal_fft.shape) > 1:
        print("WARNING: Ensure that frequency is the final axis")

    n = signal_fft.shape[-1]
    amp = np.abs(signal_fft) / n

    freq_limit_index = int(math.floor(n / 2))
    pos_amp = 2 * np.take(amp, range(1, freq_limit_index), axis=-1)
    pos_freqs = np.take(freqs, range(1, freq_limit_index), axis=-1)

    if scaling == 'amplitude':
        result = pos_amp
    elif scaling == 'power':
        result = (pos_amp) ** 2
    elif scaling == 'R2':
        result = ((n / 2) * (pos_amp ** 2)) / ((n - 1) * (variance))

    return result, pos_freqs


def inverse_fourier_transform(coefficients, sample_freq, min_freq=None, max_freq=None, exclude='negative'):
    assert exclude in ['positive', 'negative', None]
    coefs = deepcopy(coefficients)
    if exclude == 'positive':
        coefs[sample_freq > 0] = 0
    elif exclude == 'negative':
        coefs[sample_freq < 0] = 0

    if (max_freq == min_freq) and max_freq:
        coefs[np.abs(sample_freq) != max_freq] = 0

    if max_freq:
        coefs[np.abs(sample_freq) > max_freq] = 0

    if min_freq:
        coefs[np.abs(sample_freq) < min_freq] = 0

    return fftpack.ifft(coefs)
