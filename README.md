# Non-Invasive Detection of Coronary Artery Disease Using Photoplethysmography and Machine Learning

## 📌 Project Overview
Coronary Artery Disease (CAD) is one of the leading causes of mortality worldwide. Early diagnosis plays a crucial role in preventing severe cardiovascular events such as myocardial infarction and heart failure. Traditional diagnostic techniques like coronary angiography and ECG-based stress tests are effective but often invasive, expensive, and unsuitable for large-scale screening.

This project proposes a **non-invasive, machine learning–based CAD detection system** using **Photoplethysmography (PPG) signals**. By extracting clinically relevant features from PPG waveforms and applying supervised learning models, the system aims to provide a cost-effective and scalable screening solution.

---

## 🎯 Objectives
- Develop a **non-invasive CAD detection pipeline** using PPG signals
- Extract **time-domain, frequency-domain, and morphological features**
- Train and evaluate multiple **machine learning classifiers**
- Ensure **robust validation and reproducibility**
- Provide a **clean ML engineering workflow** suitable for research and deployment

---

## 🧠 Methodology Overview
1. **Data Acquisition**
   - PPG signals collected from subjects with and without CAD
   - Data provided in CSV format

2. **Preprocessing**
   - Noise handling and signal normalization
   - Removal of missing and corrupted samples
   - Feature harmonization across datasets

3. **Feature Extraction**
   - Time-domain features
   - Frequency-domain features
   - Pulse morphology features

4. **Model Training**
   - Logistic Regression
   - Random Forest
   - Gradient-based ensemble models

5. **Evaluation**
   - Stratified K-Fold Cross-Validation
   - ROC-AUC, Precision, Recall, F1-score
   - Precision–Recall analysis

6. **Inference**
   - CAD risk prediction on unseen PPG samples

---

## 🗂️ Project Structure
```text
CAD_Project/
├── data/
│   ├── raw/                  # Original PPG CSV files (ignored by git)
│   ├── processed/            # Feature-extracted datasets
│   └── inference/            # Input data for prediction
│
├── models/
│   ├── CAD_Harmonized_91_77_Model.pkl
│
├── src/
│   ├── preprocessing.py      # Signal cleaning & normalization
│   ├── feature_extraction.py # CAD-related PPG feature extraction
│   ├── train.py              # Model training pipeline
│   ├── evaluate.py           # Model evaluation
│   └── predict.py            # Inference script
│
├── logs/                     # Training & evaluation logs
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```
---

## 🧪 Feature Extraction Details

Feature extraction plays a crucial role in capturing cardiovascular characteristics from PPG signals that are indicative of Coronary Artery Disease (CAD). Since CAD affects arterial stiffness, blood flow, and vascular compliance, both temporal and morphological changes in PPG waveforms are clinically significant.

The extracted features are grouped into three categories: time-domain, frequency-domain, and morphological features.

---

### 🔹 Time-Domain Features
Time-domain features capture statistical and temporal variations in the PPG signal and reflect heart rate dynamics and pulse regularity.

- Mean, variance, and standard deviation of the signal  
- Skewness and kurtosis to measure waveform asymmetry  
- Pulse-to-pulse interval  
- Heart Rate Variability (HRV)–related indicators  

These features help quantify irregularities in cardiac rhythm and pulse dynamics commonly observed in CAD patients.

---

### 🔹 Frequency-Domain Features
Frequency-domain features analyze the spectral content of PPG signals, providing insights into autonomic nervous system activity and vascular regulation.

- Power Spectral Density (PSD)  
- Dominant frequency components  
- Low-Frequency (LF) and High-Frequency (HF) band power  

Alterations in frequency components are associated with impaired vascular function and reduced arterial compliance in CAD.

---

### 🔹 Morphological Features (CAD-Relevant)
Morphological features describe the shape and structure of individual PPG pulses, which are highly sensitive to arterial stiffness and blood flow resistance.

- Pulse amplitude  
- Systolic peak time  
- Diastolic decay time  
- Augmentation index  
- Pulse width measured at different amplitude levels  

Changes in these features are directly linked to vascular abnormalities caused by atherosclerosis and coronary artery narrowing.
Overall, the extracted feature set provides a comprehensive representation of cardiovascular health and serves as an effective input for machine learning–based CAD classification.

---



## 🤖 Machine Learning Models Used
**1. Logistic Regression**

- Baseline interpretable model
- Provides probabilistic CAD risk estimates
- Useful for clinical explainability

**2. Random Forest Classifier**
- Handles non-linear feature interactions
- Robust to noise and overfitting
- Provides feature importance analysis

**3. Ensemble / Gradient-Based Models**
-Improves classification performance
- Captures complex physiological patterns
- Optimized using cross-validation

## 📊 Evaluation Metrics

**ROC-AUC Score** – Overall discrimination ability

**Precision** – Accuracy of CAD predictions

**Recall (Sensitivity)** – Ability to detect CAD cases

**F1-Score** – Balance between precision and recall

**Average Precision Score** – Performance under class imbalance

## 🚀 How to Run the Project
**1️⃣ Create Virtual Environment**
```text
python -m venv env
source env/bin/activate   # Linux / Mac
env\Scripts\activate      # Windows
```
**2️⃣ Install Dependencies**
```text
pip install -r requirements.txt
```
**3️⃣ Train the Model**
```text
python src/train.py
```
**4️⃣ Evaluate the Model**
```text
python src/evaluate.py
```

**5️⃣ Run Inference**
```text
python src/predict.py --mode evaluate --input data/inference/cleaned-test-ppg-data.csv
```

## 📈 Key Challenges & Solutions
| Challenge           | Solution                           |
| ------------------- | ---------------------------------- |
| Noisy PPG signals   | Signal normalization and filtering |
| Class imbalance     | Stratified sampling & PR metrics   |
| Feature variability | Feature harmonization              |
| Overfitting         | Cross-validation & ensemble models |

## 🔬 Results Summary

- Achieved high ROC-AUC and recall, suitable for screening use cases
- Random Forest and ensemble models outperformed linear baselines
- Demonstrated feasibility of PPG-based CAD detection

## 🧾 Dependencies
```text
Python 3.8+
NumPy
Pandas
Scikit-learn
SciPy
Matplotlib
```
(See requirements.txt for the full list)

## 📌 Future Work

- Deep learning on raw PPG waveforms
- Explainable AI (SHAP / LIME)
- Real-time wearable integration
- Multi-center dataset validation

## 📜 License

This project is intended for academic and research purposes only.

## 🙌 Acknowledgements

- Open-source ML community
- Academic literature on PPG-based cardiovascular analysis
- Clinical insights into CAD physiology
