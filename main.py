import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from signals import exponential_signal, sinusoid_signal, linear_chirp

# poly smoothness is a function of encoder resolution.
# poly overshoot is not a function of encoder resolution.
encoder_resolution = 1024
time_step = 0.001
t0 = 0
tf = 3.

# filter variables
poly_window = 9
degree = 2
sigma = 5

use_exponential = False
use_sinusoid = False
use_chirp = True

add_noise = True

estimator_types = ['dx / dt', 'Var. dx / dt', 'Poly',
                   'Poly Sigma', 'Poly Sigma Shift', 'Sav-Gol']

show_types = [
    # estimator_types[0],
    # estimator_types[1],
    # estimator_types[2],
    estimator_types[3],
    estimator_types[4],
    # estimator_types[5]
]

time = np.arange(t0, tf, time_step)
if use_exponential:
    y, y_d, y_dd = exponential_signal(time, gamma=-3.0, y0=1.0)
elif use_sinusoid:
    y, y_d, y_dd = sinusoid_signal(time, A=1.0, w=4.0 * np.pi, phase=0.0)
elif use_chirp:
    max_w = 3 * np.pi
    y, y_d, y_dd = linear_chirp(time, A=1.0, max_w=max_w, min_w=0.0, phase=0.0)

if add_noise:
    noise = np.random.normal(loc=0.0, scale=0.01, size=y.shape)
    y = y + noise

cntr_y = np.asarray((y * encoder_resolution), dtype=np.int32)


def dx_dt(time, y):
    """
    Calculates velocity and acceleration using the traditional method of finite
    differences.
    """
    y_d = np.zeros(time.shape)
    y_dd = np.zeros(time.shape)

    y_d[1:] = np.diff(y) / time_step
    y_dd[1:] = np.diff(y_d) / time_step

    y_d = y_d / encoder_resolution
    y_dd = y_dd / encoder_resolution

    return y_d, y_dd


def var_dx_dt(time, y):
    """
    Calculates velocity and acceleration using a variable approach. If there is no
    change in the count variable then increment a the time over which the velocity is taken.
    This helps at low speeds ove dx_dt but as a result can never produce a derivative value of 0.
    """
    y_d = np.zeros(time.shape)
    y_dd = np.zeros(time.shape)

    cnt = 1
    dy = np.diff(y)
    for i in range(1, time.shape[0]):
        if dy[i - 1] == 0:
            y_d[i] = y_d[i - 1]
            cnt += 1
        else:
            y_d[i] = dy[i - 1] / (cnt * time_step)
            cnt = 1

    cnt = 1
    dyd = np.diff(y_d)
    for i in range(2, time.shape[0]):
        if dyd[i - 1] == 0:
            y_dd[i] = y_dd[i - 1]
            cnt += 1
        else:
            y_dd[i] = dyd[i - 1] / (cnt * time_step)
            cnt = 1

    y_d = y_d / encoder_resolution
    y_dd = y_dd / encoder_resolution

    return y_d, y_dd


def poly(time, y):
    """
    Calculates velocity and acceleration by fitting a polynomial over the prior N elements
    defined by poly_window.
    """
    y_d = np.zeros(time.shape)
    y_dd = np.zeros(time.shape)

    for i in range(poly_window, time.shape[0]):
        params = np.polyfit(x=time[i - poly_window + 1:i + 1],
                            y=y[i - poly_window + 1:i + 1], deg=degree)
        p = np.poly1d(params)
        p_d = np.polyder(p)
        p_dd = np.polyder(p_d)
        y_d[i] = p_d(time[i])
        y_dd[i] = p_dd(time[i])

    y_d = y_d / encoder_resolution
    y_dd = y_dd / encoder_resolution

    return y_d, y_dd


def poly_sigma(time, y):
    """
    Calculates velocity and acceleration by fitting a polynomial over the prior N elements
    defined by poly_window. The sigma value defines an averaging window inside of poly_window
    in which all points inside of sigma window are averaged and treated as a single point for the
    polynomial fitting process.
    """
    y_d = np.zeros(time.shape)
    y_dd = np.zeros(time.shape)
    window = poly_window

    for i in range(window * sigma, time.shape[0]):
        y_history = y[i - window * sigma + 1:i + 1]
        y_hist_avg = np.mean(y_history.reshape(-1, sigma), axis=1)
        params = np.polyfit(
            x=time[i - window * sigma + 1:i + 1:sigma], y=y_hist_avg, deg=degree)

        p = np.poly1d(params)
        p_d = np.polyder(p)
        p_dd = np.polyder(p_d)

        y_d[i] = p_d(time[i])
        y_dd[i] = p_dd(time[i])

    y_d = y_d / encoder_resolution
    y_dd = y_dd / encoder_resolution

    return y_d, y_dd


degrees = []
def poly_sigma_shift(time, y):
    """
    Calculates velocity and acceleration using the poly_sigma method but includes
    extra logic for selecting which degree the polynomial should be.
    """
    y_d = np.zeros(time.shape)
    y_dd = np.zeros(time.shape)
    window = poly_window

    for i in range(window * sigma, time.shape[0]):
        if abs(y_d[i-1]) < 10*encoder_resolution:
            degree = 2
        else:
            degree = 3

        degrees.append(degree)

        # if abs(y_d[i - 1]) < 3:
        #     degree = 2
        # elif abs(y_d[i - 1]) > 3 and abs(y_dd[i - 1]) < 0.2:
        #     degree = 2
        # else:
        #     degree = 3

        y_history = y[i - window * sigma + 1:i + 1]
        y_hist_avg = np.mean(y_history.reshape(-1, sigma), axis=1)

        params = np.polyfit(
            x=time[i - window * sigma + 1:i + 1:sigma], y=y_hist_avg, deg=degree)

        p = np.poly1d(params)
        p_d = np.polyder(p)
        p_dd = np.polyder(p_d)

        y_d[i] = p_d(time[i])
        y_dd[i] = p_dd(time[i])

    y_d = y_d / encoder_resolution
    y_dd = y_dd / encoder_resolution

    return y_d, y_dd


def savgol(time, y):
    """
    Calculates velocity and acceleration using savgol filter from scipy. Since
    the filter assumes that the whole signal is known, I perform differentiation
    on the final two points after savgol is applied.
    """
    y_d = np.zeros(time.shape)
    y_dd = np.zeros(time.shape)
    window = poly_window

    for i in range(window * sigma, time.shape[0]):
        y_history = y[i - window * sigma + 1:i + 1]
        filtered_values = signal.savgol_filter(
            y_history, window_length=window, polyorder=degree)
        y_d[i] = (filtered_values[-1] - filtered_values[-2]) / time_step

    for i in range(window * sigma, time.shape[0]):
        y_history = y_d[i - window * sigma + 1:i + 1]
        filtered_values = signal.savgol_filter(
            y_history, window_length=window, polyorder=degree)
        y_dd[i] = (filtered_values[-1] - filtered_values[-2]) / time_step

    y_d = y_d / encoder_resolution
    y_dd = y_dd / encoder_resolution

    return y_d, y_dd

differentation_types = {
    'dx / dt': dx_dt,
    'Var. dx / dt': var_dx_dt,
    'Poly': poly,
    'Poly Sigma': poly_sigma,
    'Poly Sigma Shift': poly_sigma_shift,
    'Sav-Gol': savgol,
}


# plt.ion()
# PLOT SIGNAL
fig = plt.figure()
plt.plot(time, y, label='True')
plt.plot(time, cntr_y * 1.0 / encoder_resolution, label='Encoder')

plt.title('System with Encoder (resolution, dt) = %d, %.3f' %
          (encoder_resolution, time_step))
plt.xlabel('Time (s)')
plt.ylabel('Encoder Counter (ticks)')
plt.legend()
plt.show()

# PLOT VELOCITY
fig = plt.figure()
plt.plot(time, y_d, label='True Speed')

for est_type in show_types:
    y_d, null = differentation_types[est_type](time, cntr_y)
    plt.plot(time, y_d, label=est_type)

plt.title('Velocity of System with Encoder (resolution, dt) = %d, %.3f' %
          (encoder_resolution, time_step))
plt.xlabel('Time (s)')
plt.ylabel('Rev / Second')
plt.legend()
plt.show()


fig = plt.figure()
plt.plot(time, y_dd, label='True Acc')

for est_type in show_types:
    y_d, y_dd = differentation_types[est_type](time, cntr_y)
    plt.plot(time, y_dd, label=est_type)

plt.title('Acceleration of System with encoder (resolution, dt) = %d, %.3f' %
          (encoder_resolution, time_step))
plt.xlabel('Time (s)')
plt.ylabel('Rev / Second^2')
plt.legend()
plt.show()
