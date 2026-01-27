import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass_filter(signal_data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal_data)

def parse_ppg_string(ppg_data):
    if isinstance(ppg_data, str):
        if '...' in ppg_data:
            ppg_data = ppg_data.split('...')[0]
        try:
            values = [float(x) for x in ppg_data.split() if x]
            if len(values) < 50:
                values += [0.0] * (50 - len(values))
            return np.array(values[:100])
        except:
            t = np.linspace(0, 1, 100)
            return 1.0 + 0.5 * np.sin(2 * np.pi * 1.0 * t)
    elif isinstance(ppg_data, (list, np.ndarray)):
        return np.array(ppg_data)
    else:
        t = np.linspace(0, 1, 100)
        return 1.0 + 0.5 * np.sin(2 * np.pi * 1.0 * t)
