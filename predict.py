import numpy as np
import tensorflow as tf
import keras
from scipy.io import wavfile
from sklearn import preprocessing
import librosa
from scipy.stats import mode
# Load the trained model
model = keras.models.load_model('voice_gender_model.h5')

def preprocess_audio(audio_file):
    # Load audio file and extract features (adjust based on your actual feature extraction method)
    y, sr = librosa.load(audio_file)
    # Extract features (adjust based on your actual feature extraction method)
    features = extract_features(y, sr)
    # Normalize the features using the same scaler used during training
    features_normalized = preprocessing.MinMaxScaler().fit_transform(features)
    return features_normalized

def extract_features(y, sr):
    # Placeholder, replace with your actual feature extraction code
    meanfreq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    sd = np.std(librosa.feature.mfcc(y=y, sr=sr))
    median = np.median(librosa.feature.mfcc(y=y, sr=sr))
    
    # Additional features
    q25 = np.percentile(librosa.feature.mfcc(y=y, sr=sr), 25)
    q75 = np.percentile(librosa.feature.mfcc(y=y, sr=sr), 75)
    iqr = q75 - q25
    skew = librosa.feature.spectral_bandwidth(y=y, sr=sr, p=3)  # Adjust p value if needed
    kurt = librosa.feature.spectral_bandwidth(y=y, sr=sr, p=4)  # Adjust p value if needed
    sp_ent = np.mean(librosa.feature.spectral_flatness(y=y))
    sfm = np.mean(librosa.feature.spectral_flatness(y=y))
    
    hist, edges = np.histogram(y, bins=np.arange(256))
    mode_value = edges[np.argmax(hist)]
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    meanfun = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    minfun = np.min(librosa.feature.mfcc(y=y, sr=sr))
    maxfun = np.max(librosa.feature.mfcc(y=y, sr=sr))
    meandom = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    mindom = np.min(librosa.feature.mfcc(y=y, sr=sr))
    maxdom = np.max(librosa.feature.mfcc(y=y, sr=sr))
    dfrange = maxdom - mindom
    modindx = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    features=np.array([meanfreq, sd, median, q25, q75, iqr, skew, kurt, sp_ent, sfm, mode_value, centroid, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx])
    print(features)
    # Return the features as a numpy array
    return features


def predict_gender(audio_file):
    # Preprocess the audio file
    processed_audio = preprocess_audio(audio_file)

    # Make predictions
    predictions = model.predict(processed_audio)

    # Assuming binary classification (male/female)
    gender = "Male" if predictions[0][0] > 0.5 else "Female"

    return gender

if __name__ == "__main__":
    # Replace 'your_audio_file.wav' with the path to the user's voice file
    user_audio_file = 'brian.wav'
    
    predicted_gender = predict_gender(user_audio_file)

    print(f"Predicted Gender: {predicted_gender}")
