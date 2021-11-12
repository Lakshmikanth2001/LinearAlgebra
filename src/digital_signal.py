import numpy as np


def fft(x):
    """
    A recursive implementation of
    the 1D Cooley-Tukey FFT, the
    input should have a length of
    power of 2.
    """
    N = len(x)

    if N == 1:
        return x
    else:
        xk_even = fft(x[::2])
        xk_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)

        xk = np.concatenate(
            [xk_even + factor[:int(N / 2)] * xk_odd,
             xk_even + factor[int(N / 2):] * xk_odd])
        return xk


def dft(x):
    """
    Function to calculate the discrete Fourier Transform of a 1D real-valued signal x
    """
    n_array = len(x)
    n = np.arange(n_array)
    k = n.reshape((n_array, 1))
    e = np.exp(-2j * np.pi * k * n / n_array)

    xk = np.dot(e, x)
    return xk


def inverse_dft(X, signal_nature='real'):
    """
    Function to calculate the inverse discrete Fourier Transform of a 1D real-valued signal x
    """
    n = len(X)

    k = np.arange(0, n).reshape(n, 1)
    n_array = np.arange(0, n).reshape(1, n)

    matrix = np.exp(2j * np.pi * k * n_array / n)

    if signal_nature == 'real':
        return np.real(np.dot(matrix, X) / n)
    elif signal_nature == 'imaginary':
        return 1j * np.imag(np.dot(matrix, X) / n)
    else:
        return np.dot(matrix, X) / n
