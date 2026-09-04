# Parkinson
Detecting Parkinson Disease


# Origin of the Parkinson's Disease Speech Signal Features Dataset

The Parkinson’s disease dataset containing **755 columns and 756 rows** is known as the **Parkinson's Disease (PD) Speech Signal Features Dataset** (commonly distributed as `pd_speech_features.csv`). 

## Origin and Collection
* **Institution:** Department of Neurology, Cerrahpaşa Faculty of Medicine, Istanbul University
* **Repository:** [UCI Machine Learning Repository](https://uci.edu)
* **Donation Year:** 2018
* **Lead Researchers:** C. Okan Sakar et al.

## Study Participants (756 Rows)
The data was collected from **252 total subjects**:
* **188 patients** diagnosed with Parkinson's Disease
* **64 healthy individuals** serving as the control group

During clinical examinations, a 44.1 kHz microphone recorded each subject performing **three repetitions of sustained phonation of the vowel /a/**. Because each of the 252 subjects recorded three distinct times, the dataset contains exactly **756 rows** (252 × 3).

## Feature Structure (755 Columns)
A variety of speech signal processing algorithms were applied to the audio files to extract **754 predictive features**, while the final column serves as the target variable. 

The columns are structured as follows:
* **Subject metadata (Columns 1–2):** Subject Identifier (`id`) and categorical gender feature.
* **Baseline Features (Columns 3–23):** Standard voice metrics like jitter, shimmer, fundamental frequency parameters, Detrended Fluctuation Analysis (DFA), and Pitch Period Entropy (PPE).
* **Acoustic Parameters (Columns 24–34):** Statistical intensity parameters, formant frequencies, and bandwidth properties.
* **Vocal Fold Features (Columns 35–56):** Metrics characterizing physical vocal fold vibration models.
* **MFCCs (Columns 57–140):** Mel Frequency Cepstral Coefficients that capture the envelope of the short-term power spectrum.
* **Wavelet Features (Columns 141–322):** Wavelet transform data representing time-frequency variations.
* **TQWT Features (Columns 323–754):** Tunable Q-factor Wavelet Transform features, which make up the bulk of the high-dimensional attributes to deeply quantify frequency deviations.
* **Class (Column 755):** The final classification column (Target Variable: 0 for healthy, 1 for PD).

