import numpy as np
def exponential_signal(time, gamma=-5.0, y0=1.0):
    """
    https://en.wikipedia.org/wiki/Exponential_decay

    :type time: np.ndarray
    :param time: Time array

    :type gamma: float
    :param gamma: Exponential rate

    :type y0: float
    :param y0: Initial value
    """
    y = y0 * np.exp(gamma * time)
    y_d = y0 * gamma * np.exp(gamma * time)
    y_dd = y0 * (gamma ** 2) * np.exp(gamma * time)

    return y, y_d, y_dd


def sinusoid_signal(time, A=1.0, w=np.pi, phase=0.0):
    """
    https://en.wikipedia.org/wiki/Sine_wave

    :type time: np.ndarray
    :param time: Time array

    :type A: float
    :param A: Amplitude

    :type w: float
    :param w: Angular rate

    :type phase: float
    :param phase: Phase value
    """
    y = A * np.sin(time * w + phase)
    y_d = w * A * np.cos(time * w + phase)
    y_dd = -w ** 2 * A * np.sin(time * w + phase)

    return y, y_d, y_dd


def linear_chirp(time, A=1.0, max_w=4*np.pi, min_w=0.0, phase=0.0):
    """
    https://en.wikipedia.org/wiki/Chirp

    :type time: np.ndarray
    :param time: Time array

    :type A: float
    :param A: Amplitude

    :type max_w: float
    :param max_w: Maximum Angular rate

    :type phase: float
    :param phase: Phase value
    """
    w = np.arange(min_w, max_w, step=max_w / time.shape[0])
    y = A * np.sin(time * w + phase)
    y_d = (w + max_w * time / time[-1]) * A * np.cos(time * w + phase)
    y_dd = np.zeros(time.shape)
    y_dd[1:] = np.diff(y_d) / (time[1]-time[0])

    return y, y_d, y_dd
