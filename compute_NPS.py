import numpy as np
from neurodsp.spectral import compute_spectrum_welch
from fooof import FOOOF


def compute_NPS(data):

    n_trial = data.shape[0]
    n_channel = data.shape[1]
    freq_range = [0.5, 35]
    NPS = np.empty([n_trial, n_channel, 6, 4])

    for trial in range(n_trial):
        for channel in range(n_channel):
            signal = data[trial, channel, :]
            freqs, spectrum =  compute_spectrum_welch(signal, 250, f_range = [0.5, 35], noverlap = 200)
            fm = FOOOF(aperiodic_mode='knee', peak_width_limits=[0,1], min_peak_height=0.05)
            fm.fit(freqs, spectrum, freq_range)
            ap_parameter =  np.array(fm.aperiodic_params_)
            p_parameter =  np.array(fm.peak_params_)

            NPS[trial, channel, :, 0] =  np.append(np.array([0, 0, 0]), ap_parameter)
            NPS[trial, channel, :, 1] =  np.append(np.array([0, 0, 0]), ap_parameter)
            NPS[trial, channel, :, 2] =  np.append(np.array([0, 0, 0]), ap_parameter)
            NPS[trial, channel, :, 3] =  np.append(np.array([0, 0, 0]), ap_parameter)

            for i in range(len(p_parameter)):

                # if (p_parameter[i][0]>=0 and p_parameter[i][0]<4) and p_parameter[i][1] > NPS[trial, channel, 2, 0]:
                #     NPS[trial, channel, 0:3 , 0] = p_parameter[i][:]
                # elif (p_parameter[i][0]>=4 and p_parameter[i][0]<8) and p_parameter[i][1] > NPS[trial, channel, 2, 1]:
                #     NPS[trial, channel, 0:3 , 1] = p_parameter[i][:]
                if (p_parameter[i][0]>=8 and p_parameter[i][0]<13) and p_parameter[i][1] > NPS[trial, channel, 2, 2]:
                    for j in range(4):
                        NPS[trial, channel, 0:3 , j] = p_parameter[i][:]
                # elif (p_parameter[i][0]>=13 and p_parameter[i][0]<=30) and p_parameter[i][1] > NPS[trial, channel, 2, 3]:
                #     NPS[trial, channel, 0:3 , 3] = p_parameter[i][:]

    return NPS


def normalize_NPS(NPS):
    # Check for inf or nan values in the data
    if np.any(np.isnan(NPS)) or np.any(np.isinf(NPS)):
        # Handle the inf and nan values by setting them to 0
        mean_NPS = np.nanmean(NPS, axis=0)
        NPS[np.isnan(NPS) | np.isinf(NPS)] = mean_NPS

    # Calculate mean and standard deviation along axis 0 (n_trial)
    mean_NPS = np.mean(NPS, axis=0)
    std_NPS = np.std(NPS, axis=0)

    # Add a small constant to avoid division by zero
    epsilon = 1e-10
    std_NPS = np.where(std_NPS == 0, epsilon, std_NPS)

    # Apply Z-score normalization along axis 0
    normalized_NPS = (NPS - mean_NPS) / std_NPS

    return normalized_NPS