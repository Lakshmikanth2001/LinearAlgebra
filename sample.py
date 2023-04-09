import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from src.digital_signal import dft

    x = np.arange(0, 10, 0.001)

    y = [np.sin(2*np.pi*50*i) for i in x]

    X = dft(y)

    X = np.abs(X)

    plt.plot(X)

    plt.show()