import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.stats import skew
from .utils import butter_bandpass_filter, parse_ppg_string

class HarmonizedPPGFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, sampling_rate=100, n_features_to_keep=20):
        self.sampling_rate = sampling_rate
        self.n_features_to_keep = n_features_to_keep
        self.selected_feature_indices_ = None
        

    def extract_single_ppg_features(self, ppg_signal):
        if len(ppg_signal) < 20:
            return np.zeros(10)

        try:
            ppg_filtered = butter_bandpass_filter(
                ppg_signal, 0.5, 8.0, self.sampling_rate
            )
        except:
            ppg_filtered = ppg_signal

        peaks, _ = find_peaks(
            ppg_filtered,
            distance=max(self.sampling_rate // 3, 10),
            prominence=0.05,
            width=2
        )

        crest_time = peaks[0] / len(ppg_filtered) if len(peaks) > 0 else 0.15
        stiffness_idx = (
            ppg_filtered[peaks[0]] / (crest_time + 1e-10)
            if len(peaks) > 0 else 5.0
        )

        heart_rate = 72.0
        if len(peaks) >= 2:
            rr = np.diff(peaks) / self.sampling_rate
            heart_rate = 60 / np.mean(rr)

        hrv = np.log1p(50)
        if len(peaks) >= 3:
            hrv = np.log1p(np.std(rr) * 1000)

        fft_vals = fft(ppg_filtered)
        mag = np.abs(fft_vals[:len(ppg_filtered)//2])
        freqs = fftfreq(len(ppg_filtered), 1/self.sampling_rate)[:len(ppg_filtered)//2]

        hr_band = (freqs >= 0.5) & (freqs <= 4.0)
        spectral_power = np.log1p(np.sum(mag[hr_band]**2)) if np.any(hr_band) else np.log1p(100)
        dominant_freq = freqs[hr_band][np.argmax(mag[hr_band])] if np.any(hr_band) else 1.2

        return np.array([
            crest_time,
            min(stiffness_idx, 50),
            0.4,
            min(max(heart_rate, 40), 180),
            hrv,
            spectral_power,
            dominant_freq,
            np.mean(ppg_filtered),
            np.std(ppg_filtered),
            skew(ppg_filtered)
        ])

    def fit_transform(self, X, y):
        features = self.transform(X)

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )

        rf.fit(features, y)
        importances = rf.feature_importances_
        all_names = self.get_all_feature_names()

        self.selected_feature_indices_ = np.argsort(importances)[::-1][:self.n_features_to_keep]
        all_names = self.get_all_feature_names()

        self.selected_feature_names_ = [
            all_names[i] for i in self.selected_feature_indices_
        ]

        return features[:, self.selected_feature_indices_]


    def transform(self, X):
        processed = []

        for idx in range(len(X)):
            ppg_features = []
            for col in [
                'ppg_min_3','ppg_min_4','ppg_min_5',
                'ppg_min_6','ppg_min_7','ppg_min_8'
            ]:
                signal = parse_ppg_string(X[col].iloc[idx])
                ppg_features.extend(self.extract_single_ppg_features(signal))

            age_norm = X['AGE'].iloc[idx] / 100.0
            gender = 1 if X['GENDER'].iloc[idx] == 'M' else 0
            diabetic = X['is_diabetic'].iloc[idx]
            cholesterol = X['has_high_cholesterol'].iloc[idx]
            obese = X['is_obese'].iloc[idx]

            interactions = [
                age_norm * diabetic,
                age_norm * cholesterol,
                diabetic * cholesterol,
                age_norm * ppg_features[1],
                cholesterol * ppg_features[1],
                ppg_features[0] * age_norm,
                ppg_features[0] * cholesterol,
                age_norm * diabetic * cholesterol
            ]

            row = [age_norm, gender, diabetic, cholesterol, obese] + interactions + ppg_features
            processed.append(row)

        X_arr = np.array(processed)
        return (
            X_arr[:, self.selected_feature_indices_]
            if self.selected_feature_indices_ is not None
            else X_arr
        )
    
    def get_all_feature_names(self):
        base = [
            "age_norm", "gender", "is_diabetic",
            "has_high_cholesterol", "is_obese"
        ]

        interactions = [
            "age_diabetic",
            "age_cholesterol",
            "diabetic_cholesterol",
            "age_stiffness",
            "cholesterol_stiffness",
            "crest_age",
            "crest_cholesterol",
            "age_diabetic_cholesterol"
        ]

        ppg_features = []
        for i in range(6):  # ppg_min_3 to ppg_min_8
            ppg_features.extend([
                f"ppg{i}_crest_time",
                f"ppg{i}_stiffness",
                f"ppg{i}_dummy",
                f"ppg{i}_heart_rate",
                f"ppg{i}_hrv",
                f"ppg{i}_spectral_power",
                f"ppg{i}_dominant_freq",
                f"ppg{i}_mean",
                f"ppg{i}_std",
                f"ppg{i}_skew"
            ])

        return base + interactions + ppg_features


    def fit(self, X, y=None):
        return self
