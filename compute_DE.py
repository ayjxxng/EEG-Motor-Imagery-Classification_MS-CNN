import math
import numpy as np
from preprocessing import butter_bandpass_filter


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def decompose(data):
    # trial*channel*sample
    start_index = 0 # 3s pre-trial signals
    shape = data.shape
    frequency = 250

    decomposed_de = np.empty([shape[0], 4, 3, 8])

    for trial in range(shape[0]):
        for channel in range(3):
            trial_signal = data[trial, channel,:]

            # ****************compute base DE****************

            delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 13, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 13, 30, frequency, order=3)

            DE_delta = np.zeros((1,8), dtype=float)
            DE_theta = np.zeros((1,8), dtype=float)
            DE_alpha = np.zeros((1,8), dtype=float)
            DE_beta = np.zeros((1,8), dtype=float)

            DE_delta = np.array([compute_DE(delta[index * 125:(index + 1) * 125], 1, 4) for index in range(8)])
            DE_theta = np.array([compute_DE(theta[index * 125:(index + 1) * 125], 4, 8) for index in range(8)])
            DE_alpha = np.array([compute_DE(alpha[index * 125:(index + 1) * 125], 8, 13) for index in range(8)])
            DE_beta = np.array([compute_DE(beta[index * 125:(index + 1) * 125], 13, 30) for index in range(8)])

            decomposed_de[trial, 0, channel, :] = DE_delta
            decomposed_de[trial, 1, channel, :] = DE_theta
            decomposed_de[trial, 2, channel, :] = DE_alpha
            decomposed_de[trial, 3, channel, :] = DE_beta

    decomposed_de = decomposed_de.transpose([0,2,3,1])

    print("trial_DE shape:", decomposed_de.shape) # output.shape = {120,4,3,8}
    return decomposed_de


def normalize_DE(decomposed_de):
    # Check for inf or nan values in the data
    if np.any(np.isnan(decomposed_de)) or np.any(np.isinf(decomposed_de)):
        # Handle the inf and nan values by setting them to 0
        mean_de = np.nanmean(decomposed_de, axis=0)
        decomposed_de[np.isnan(decomposed_de) | np.isinf(decomposed_de)] = mean_de

    # Calculate mean and standard deviation along axis 0 (n_trial)
    mean_de = np.mean(decomposed_de, axis=0)
    std_de = np.std(decomposed_de, axis=0)

    # Add a small constant to avoid division by zero
    epsilon = 1e-10
    std_de = np.where(std_de == 0, epsilon, std_de)

    # Apply Z-score normalization along axis 0
    normalized_de = (decomposed_de - mean_de) / std_de

    return normalized_de