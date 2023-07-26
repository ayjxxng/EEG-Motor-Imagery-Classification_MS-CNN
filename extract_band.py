pip install neurodsp

# Imports from NeuroDSP to simulate & plot time series
from neurodsp.sim import sim_powerlaw, set_random_seed
from neurodsp.filt import filter_signal
from neurodsp.plts import plot_time_series
from neurodsp.utils import create_times

def extract_bands(xx):
  '''
  xx={trials x channels X signals} 형태
  '''
  trials = xx.shape[0]
  channels = xx.shape[1]
  signals = xx.shape[2]
  band_sig = np.zeros((trials, 4, channels, signals))
  s_rate = 250

  for i in range(trials):
    for j in range(channels):
      band_sig_delta = filter_signal(xx[i][j], s_rate, 'bandpass', [1, 4])
      band_sig_theta = filter_signal(xx[i][j], s_rate, 'bandpass', [4, 8])
      band_sig_alpha = filter_signal(xx[i][j], s_rate, 'bandpass', [8, 13])
      band_sig_beta = filter_signal(xx[i][j], s_rate, 'bandpass', [13, 30])
      band_sig[i, 0, j, :] = band_sig_delta
      band_sig[i, 1, j, :] = band_sig_theta
      band_sig[i, 2, j, :] = band_sig_alpha
      band_sig[i, 3, j, :] = band_sig_beta

  return band_sig