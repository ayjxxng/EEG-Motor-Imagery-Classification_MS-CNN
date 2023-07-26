import numpy as np
import scipy.io
from scipy.signal import butter, lfilter



def butter_bandpass (lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter (signal, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, signal)
    return y


def eog_remove (signal, eeg_ch=3, eog_ch=3):
    '''
    function for removing EOG artifact
    based on *** ref ****

    signal: input in the form (eeg_ch + eog_ch)*T
            in which eeg_ch is the number of EEG channels,
            eog_ch is the number of EOG channels,
            and T is the number of time samples
    '''

    Y = signal[:-eog_ch, :] # received by EEG sensors
    N = signal[-eog_ch:, :] # EOG singal (noise)

    autoN = np.cov(N)
    covNY = np.zeros((eog_ch, eeg_ch))

    for i in range(eog_ch):
        for j in range(eeg_ch):
            cov   = np.cov(N[i:i+1, :], Y[j:j+1, :])
            covNY[i,j] = cov[0, 1]

    b = np.linalg.inv(autoN).dot(covNY)
    return b

def mat_extractor (
    mat_file_name, beg = 750, end = 1750,
    remove_eog = True, bpf_dict={'apply':True, 'fs': 250, 'lc':4, 'hc':38, 'order':6},
    channel_norm = True
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
    Classlabel = np.array(h['Classlabel'][0][0])
    s = mat_file['s']
    signal = np.array(s)

    xx = np.empty([120, 3, end-beg]) # signals
    yy = np.empty([120])              # labels

    # EOG recordings:
    if remove_eog:
        opened = signal[POS[0][0]-1:POS[0][0]-1+15000,:] # TYP=276, POS=0
        closed = signal[POS[1][0]-1:POS[1][0]+15000-1,:] # TYP=277, POS=1
        motion = signal[POS[2][0]-1:POS[2][0]+15000-1,:] # TYP=1081,1079,1078,1077, POS=2~5
        allsig = np.concatenate((opened, closed, motion),axis=0)
        b = eog_remove(allsig)

    samples = signal # whole signal of the session
    trials_POS  = np.where(TYP == 768)[0] # indices of successive trials 
    trials = POS[trials_POS-1]
    labels  = Classlabel # labels of corresponding task

    # iteration on tasks in each session #우린 120, 160이 되겠지
    for i in range(120): # i가 0부터 119
        xx = np.empty([120, 3, end-beg]) # signals

        if i < 119:
            x = samples[trials[i][0]:trials[i+1][0],:] 
        else:
            x = samples[trials[i][0]:,:]

        if remove_eog:
            x = (x[beg: end, 0: -3] - x[beg: end, -3: x.shape[1]+1].dot(b)).T
        else:
            x = x[beg: end, 0: -3]

        if bpf_dict['apply']:
            x = butter_bandpass_filter(
                signal  = x,
                lowcut  = bpf_dict['lc'],
                highcut = bpf_dict['hc'],
                fs      = bpf_dict['fs'],
                order   = bpf_dict['order']
            )
        if channel_norm: 
            x = (x - np.mean(x))/np.std(x)
        yy[i] = Classlabel[i]
        xx[i,:,:x.shape[1]] = x

    return xx, yy
