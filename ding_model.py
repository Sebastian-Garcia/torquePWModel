import numpy as np

############################
# Ding's Muscle Model 2007 #
############################

# Real values / Known values :
Tauc = 11  # (ms) Time constant controlling the rise and decay of CN for quadriceps. '''Value from Ding's experimentation''' [2]

# Arbitrary values / Different for each person / From Ding's article :
A = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle. Set to it's rested value, will change during experience time.
A_rest = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle when rested. '''Value from Ding's experimentation''' [1]
Alpha_A = -4.0 * 10 ** -7  # (s^-2) Coefficient for force-model parameter A in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau1_rest = 44.099  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges when rested. '''Value from Ding's experimentation''' [1]
Alpha_Tau1 = 2.1 * 10 ** -5  # (N^-1) Coefficient for force-model parameter tc in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau_fat = 127000  # (ms) Time constant controlling the recovery of the three force-model parameters (A,R0,tc) during fatigue. '''Value from Ding's experimentation''' [1]
Km_rest = 0.18  # (-) Sensitivity of strongly bound cross-bridges to CN when rested. '''Value from Ding's experimentation''' [1]
TauKm = 127000  # (ms) Time constant controlling the recovery of K1m during fatigue. '''Value from Ding's experimentation''' [1]
Alpha_Km = 1.9 * 10 ** -8  # (s^-1*N^-1) Coefficient for K1m and K2m in the fatigue model. '''Value from Ding's experimentation''' [1]
R0 = 5  # (-) Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli. When fatigue included : R0 = Km + 1.04. '''Value from Ding's experimentation''' [1]
CN = 0  # (-) Representation of Ca2+-troponin complex
F = 0  # (N) Instantaneous force

# Stimulation parameters :
stim_index = -1  # Stimulation index used in the x_dot function
frequency = 12.5  # (Hz) Stimulation frequency
rest_time = 1000  # (ms) Time without electrical stimulation on the muscle
active_time = 1000  # (ms) Time with electrical stimulation on the muscle
starting_time = 0  # (ms) Time when the first train of electrical stimulation start on the muscle

# Simulation parameters :
final_time = 1200  # Stop at x milliseconds
dt = 0.001  # Integration step in milliseconds


def ding_subject_parameters(number):
    Tau1 = [53.645, 22.154, 51.684, 60.601, 28.163, 54.41, 76.472, 39.516, 19.62, 34.622]
    Tau1_avg = 44.099
    Tau2 = [1, 38.559, 1, 1, 1, 30.549, 1, 62.981, 12.462, 35.668]
    Tau2_avg = 18.522
    Km = [0.159, 0.028, 0.109, 0.137, 0.189, 0.14, 0.546, 0.177, 0.092, 0.227]
    Km_avg = 0.180
    a_scale = [0.421, 0.653, 1.034, 0.492, 1.359, 0.879, 0.200, 0.416, 0.620, 0.847]
    a_scale_avg = 0.692
    pd0 = [118.357, 106.078, 76.986, 131.405, 96.285, 91.753, 60.963, 67.877, 47.752, 71.601]
    pd0_avg = 86.906
    pdt = [89.827, 35.131, 355.973, 194.138, 184.054, 89.569, 64.378, 88.884, 162.760, 119.699]
    pdt_avg = 138.441

    if number == 'average':
        Tau1 = Tau1_avg
        Tau2 = Tau2_avg
        Km = Km_avg
        a_scale = a_scale_avg
        pd0 = pd0_avg * 10 ** -3
        pdt = pdt_avg * 10 ** -3
    elif isinstance(number, str):
        print('only string average is available')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        x = [len(Tau1), len(Tau2), len(Km), len(a_scale), len(pd0), len(pdt)]
        t = number
        for i in range(len(x)):
            if t > x[i]:
                print('Subject n°', number, ' does not exist')
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if number - 1 < 0:
            print('Subject n°', number, ' does not exist')
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            Tau1 = Tau1[number - 1]
            Tau2 = Tau2[number - 1]
            Km = Km[number - 1]
            a_scale = a_scale[number - 1]
            pd0 = pd0[number - 1] * 10 ** -3
            pdt = pdt[number - 1] * 10 ** -3

    return Tau1, Tau2, Km, a_scale, pd0, pdt


def euler(dt, x, dot_fun, u, impulse_time, t):
    return x + dot_fun(x, u, impulse_time, t) * dt


def x_dot(x, u, impulse_time, t):
    CN = x[0]
    F = x[1]
    A = x[2]
    Tau1 = x[3]
    Km = x[4]
    var_sum = 0
    global stim_index

    if round(t, 5) in u:
        stim_index += 1

    if stim_index < 0:
        A = a_scale * (1 - np.exp(-(0 - pd0) / pdt))
    else:
        A = a_scale * (1 - np.exp(-(impulse_time[stim_index] - pd0) / pdt))

    Adot = np.array([-(A - A_rest) / Tau_fat + Alpha_A * F])
    Tau1dot = np.array([-(Tau1 - Tau1_rest) / Tau_fat + Alpha_Tau1 * F])
    R0 = 5
    Kmdot = np.array([-(Km - Km_rest) / Tau_fat + Alpha_Km * F])

    if stim_index < 0:
        var_sum = 0
    elif stim_index == 0:
        Ri = 1 + (R0 - 1) * np.exp(-((u[stim_index + 1] - u[stim_index]) / Tauc))
        var_sum += Ri * np.exp(-(t - (u[stim_index])) / Tauc)
    else:
        Ri = 1 + (R0 - 1) * np.exp(-((u[stim_index] - u[stim_index - 1]) / Tauc))
        var_sum += Ri * np.exp(-(t - (u[stim_index])) / Tauc)

    CNdot = np.array([(1 / Tauc) * var_sum - (CN / Tauc)])
    Fdot = np.array([A * (CN / (Km + CN)) - (F / (Tau1 + Tau2 * (CN / (Km + CN))))])

    return np.concatenate((CNdot, Fdot, Adot, Tau1dot, Kmdot), axis=0)


def perform_integration(final_time, dt, x_initial, x_dot_fun, u, impulse_time, integration_fun):
    time_vector = [0.]
    all_x = [x_initial]
    while time_vector[-1] <= final_time:
        all_x.append(integration_fun(dt, x_initial, x_dot_fun, u, impulse_time, time_vector[-1]))
        x_initial = all_x[-1]
        time_vector.append(time_vector[-1] + dt)
    time_vector = np.array(time_vector)
    all_x = np.array(all_x).transpose()
    global stim_index
    stim_index = -1
    return time_vector, all_x


def create_impulse(frequency, impulse_time, active_period, rest_period, starting_time, final_time):
    u = []
    t = starting_time
    dt = (1 / frequency) * 1000
    t_reset = 0

    while t <= final_time:
        if t_reset <= active_period:
            u.append(round(t))
        else:
            t += rest_period - 2 * dt
            t_reset = -dt
        t_reset += dt
        t += dt

    impulse_time = [impulse_time] * (len(u))
    impulse_time = np.array(impulse_time)

    return u, impulse_time


def run_ding(frequency_hz, pulse_width_ms, duration_ms, subject='average', dt_ms=0.001):
    """
    Run the Ding 2007 model and return a normalised excitation profile.

    Parameters
    ----------
    frequency_hz  : stimulation frequency (Hz)
    pulse_width_ms: pulse width (ms) — e.g. 0.150 for 150 µs
    duration_ms   : total simulation duration (ms)
    subject       : 'average' or int 1-10 for a specific Ding subject
    dt_ms         : integration timestep (ms)

    Returns
    -------
    time_s        : time array in seconds
    excitation    : normalised force [0, 1] — use as OpenSim muscle excitation
    force_n       : raw force in Newtons
    """
    global stim_index, a_scale, pd0, pdt, Tau2
    stim_index = -1

    Tau1, Tau2, Km_val, a_scale, pd0, pdt = ding_subject_parameters(subject)

    u, imp_time = create_impulse(
        frequency_hz, pulse_width_ms,
        active_period=duration_ms,
        rest_period=0,
        starting_time=0,
        final_time=duration_ms
    )

    x_initial = np.array([0.0, 0.0, A_rest, Tau1, Km_val])
    time_ms, all_x = perform_integration(duration_ms, dt_ms, x_initial, x_dot, u, imp_time, euler)

    force = all_x[1, :]
    peak  = force.max()
    excitation = force / peak if peak > 0 else force

    # Convert time from ms to seconds for OpenSim
    time_s = time_ms / 1000.0

    return time_s, excitation, force
