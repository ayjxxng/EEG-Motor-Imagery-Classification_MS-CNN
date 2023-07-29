import numpy as np
import scipy.io
from scipy.signal import butter, lfilter
from neurodsp.filt import filter_signal


def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(signal, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, signal)
    return y


def linear_interpolation(data):
    num_channels = data.shape[1]
    num_points = data.shape[0]

    interpolated_data = np.empty((num_points, num_channels))

    for channel in range(num_channels):
        channel_data = data[:, channel]
        valid_indices = np.where(~np.isnan(channel_data))[0]

        interpolated_data[:, channel] = np.interp(
            np.arange(num_points),
            valid_indices,
            channel_data[valid_indices]
        )

    return interpolated_data


def eog_remove(signal, eeg_ch=3, eog_ch=3):
    '''
    function for removing EOG artifact
    based on *** ref ****

    signal: input in the form (eeg_ch + eog_ch)*T
            in which eeg_ch is the number of EEG channels,
            eog_ch is the number of EOG channels,
            and T is the number of time samples
    '''

    Y = signal[:-eog_ch, :]  # received by EEG sensors
    N = signal[-eog_ch:, :]  # EOG singal (noise)

    autoN = np.cov(N)
    covNY = np.zeros((eog_ch, eeg_ch))

    for i in range(eog_ch):
        for j in range(eeg_ch):
            cov = np.cov(N[i:i + 1, :], Y[j:j + 1, :])
            covNY[i, j] = cov[0, 1]

    b = np.linalg.inv(autoN).dot(covNY)
    return b


def mat_extractor(
        mat_file_name, data_type, remove_eog=True,
        bpf_dict={'apply': True, 'fs': 250, 'lc': 1, 'hc': 30, 'order': 6},
        channel_norm=True
):
    '''
    function for extracting data from a single mat file and doing preprocessing on it

    data_path    : the directroy of the desired .mat file
    beg, end     : refer to the begining and end og the part of the 8-second
                   signal that the motor imagery task is done
    remove_eog   : decides whether we want to remove EOG or use raw EEG
    bpf_dict     : a dictionary for specifications of bandpass filter that we want to apply
                   it should be defined with the following keys and values
                   'apply'   : deciding whether we want to apply bpf or not (boolean)
                   'lc','hc' : lowcut and highcut of butterworth bpf (float)
                   'fs'      : sampling rate (250 for bcic-IV-2a)
                   'order'   : the order of butterworth bpf
    channel_norm : z-score normalization for each channel (boolean)
    '''
    mat_file = scipy.io.loadmat(mat_file_name)
    h = mat_file['h']
    EVENT = h['EVENT']
    TYP = np.array(EVENT[0][0][0][0][0])
    POS = np.array(EVENT[0][0][0][0][1])
    DUR = np.array(EVENT[0][0][0][0][4])
    classlabel = np.array(h['Classlabel'][0][0])
    n_trials = classlabel.shape[0]

    cue_POS = np.where((TYP == 769) | (TYP == 770) | (TYP == 783))[0]  # indices of successive trials
    trials = POS[cue_POS]

    s = mat_file['s']
    signal = np.array(s)

    if data_type == 'T':
        dey = 0
    else:
        dey = 125

    xx = np.empty([n_trials, 3, 1000])  # signals
    yy = np.empty([n_trials])  # labels


    # EOG recordings:
    if remove_eog:
        opened = signal[POS[0][0]:POS[0][0] + 15000, :]  # TYP=276, POS=0
        closed = signal[POS[1][0]:POS[1][0] + 15000, :]  # TYP=277, POS=1
        motion = signal[POS[2][0]:POS[2][0] + 15000, :]  # TYP=1081,1079,1078,1077, POS=2~5
        allsig = np.concatenate((opened, closed, motion), axis=0)
        b = eog_remove(allsig)

    for i in range(n_trials):
        x = signal[trials[i][0] + dey: trials[i][0] + dey + 1000, :]
        x = linear_interpolation(x)
        if remove_eog:
            x = (x[:, 0: -3] - x[:, -3: x.shape[1] + 1].dot(b)).T
        else:
            x = x[:, 0: -3]

        if bpf_dict['apply']:
            x = butter_bandpass_filter(
                signal=x,
                lowcut=bpf_dict['lc'],
                highcut=bpf_dict['hc'],
                fs=bpf_dict['fs'],
                order=bpf_dict['order']
            )
        if channel_norm:
            x = (x - np.mean(x)) / np.std(x)
        yy[i] = classlabel[i]
        xx[i, :, :] = x

    return xx, yy


def extract_bands(xx):
    '''
  xx={trials x channels X signals}
  '''
    trials = xx.shape[0]
    channels = xx.shape[1]
    signals = xx.shape[2]
    extracted_sig = np.zeros((trials, 4, channels, signals))
    s_rate = 250

    for i in range(trials):
        for j in range(channels):
            band_sig_delta = filter_signal(xx[i][j], s_rate, 'bandpass', [1, 4], remove_edges=False)
            band_sig_theta = filter_signal(xx[i][j], s_rate, 'bandpass', [4, 8], remove_edges=False)
            band_sig_alpha = filter_signal(xx[i][j], s_rate, 'bandpass', [8, 13], remove_edges=False)
            band_sig_beta = filter_signal(xx[i][j], s_rate, 'bandpass', [13, 30], remove_edges=False)
            extracted_sig[i, 0, j, :] = band_sig_delta
            extracted_sig[i, 1, j, :] = band_sig_theta
            extracted_sig[i, 2, j, :] = band_sig_alpha
            extracted_sig[i, 3, j, :] = band_sig_beta

    return extracted_sig
