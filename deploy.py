import streamlit as st
import numpy as np
import joblib
import pywt
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="ECG Signal Classification",page_icon="🫀", layout="wide")

st.title("Single ECG Signal Classification")

knn = joblib.load('knn_model')

#band pass filter
def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=360, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

#normalization
def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = 2 * (signal - min_val) / (max_val - min_val) - 1
    return normalized_signal

#extract wavelet features
def extract_wavelet_features(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = np.concatenate(coeffs)   
    return features



col1, col2 = st.columns([1,1])

with col1:
    st.markdown("<h3 style='font-weight: bold; font-size: 20px;'>Enter ECG Signal (one signal, delimited by '|')</h3>", unsafe_allow_html=True)
    signal_input = st.text_area("", height=250)
    predict_button = st.button("Predict")
    if not signal_input:
        st.warning("Please enter an ECG signal to classify.")
        
with col2:
    if signal_input and predict_button:

        signal = np.array([float(i) for i in signal_input.split('|') if i.strip() != ''])

        filtered_signal = bandpass_filter(signal)

        normalized_signal = normalize_signal(filtered_signal)

        features = extract_wavelet_features(normalized_signal).reshape(1, -1)

        prediction = knn.predict(features)
     
        if prediction[0] == 0:
            st.success("The ECG signal is **Normal**.")
            st.header("Original Signal:")
            with st.spinner('Plotting the signal...'):
                 st.line_chart(signal,color='#008000')
        else:
            st.error("The ECG signal is **PVC**.")
            st.header("Original Signal:")
            with st.spinner('Plotting the signal...'):
                st.line_chart(signal,color='#FF6347')



  
