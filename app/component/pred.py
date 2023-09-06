import librosa
import numpy as np
from keras.models import load_model
import joblib
import pandas as pd
from component.Azure_Endpoint import AutoML
import warnings

warnings.filterwarnings("ignore")


genres = ['Blues', 'Classique', 'Country', 'Disco', 'Hiphop', 
          'Jazz', 'Métal', 'pop', 'Reggae', 'Rock']

column_names = ['zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3',
                'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
                'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
                'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']


# loaded_model = load_model('model/modelv1.hdf5')
loaded_model = load_model('model/modelv2.hdf5')
scaler = joblib.load('model/scalerv2.pkl') 

def audio_pipeline(audio):
    features = []

    # Calcul du ZCR

    chroma_stft = librosa.feature.chroma_stft(y=audio)
    features.append(np.mean(chroma_stft))
    features.append(np.var(chroma_stft))

    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.var(rms))

    # Calcul de la moyenne du Spectral centroid

    # spectral_centroids = librosa.feature.spectral_centroid(y=audio)[0]
    spectral_centroids = librosa.feature.spectral_centroid(y=audio)
    features.append(np.mean(spectral_centroids))
    features.append(np.var(spectral_centroids))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio)
    features.append(np.mean(spectral_bandwidth))
    features.append(np.var(spectral_bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=audio)
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features.append(np.mean(zcr))
    features.append(np.var(zcr))

    harmony = librosa.effects.harmonic(y=audio)
    features.append(np.mean(harmony))
    features.append(np.var(harmony))

    tempo = librosa.feature.tempo(y=audio)
    features.append(tempo[0])

    # Calcul des moyennes des MFCC

    mfcc = librosa.feature.mfcc(y=audio)

    for x in mfcc:
        features.append(np.mean(x))
        features.append(np.var(x))

    return features

def audio_pipelineAzure(audio):
  features = []

  # Calcul du ZCR

  zcr = librosa.zero_crossings(audio)
  features.append(sum(zcr))

  # Calcul de la moyenne du Spectral centroid

  spectral_centroids = librosa.feature.spectral_centroid(y=audio)[0]
  features.append(np.mean(spectral_centroids))

  # Calcul du spectral rolloff point

  rolloff = librosa.feature.spectral_rolloff(y=audio)
  features.append(np.mean(rolloff))

  # Calcul des moyennes des MFCC

  mfcc = librosa.feature.mfcc(y=audio)

  for x in mfcc:
    features.append(np.mean(x))


  return features

# Load and preprocess the audio file
def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=None)
    
    # Apply the same feature extraction and scaling as you did during training
    features = audio_pipeline(audio)
    scaled_features = scaler.transform([features]) 
    return scaled_features
  
def predict_ML(file_path):
  audio, _ = librosa.load(file_path, sr=None)
  features = audio_pipelineAzure(audio)
  data_frame = pd.DataFrame([features], columns=column_names)
  fromAutoml = AutoML()
  res_autoML=fromAutoml.getPred(data_frame)
  return res_autoML

# Make predictions on the preprocessed audio
def predict_top_genres(file_path, top_n=3):
    scaled_features = preprocess_audio(file_path)
    predicted_probabilities = loaded_model.predict(scaled_features)
    top_n_indices = np.argsort(predicted_probabilities[0])[::-1][:top_n]
    top_n_genres = [genres[idx] for idx in top_n_indices]
    prediction_percentages = [predicted_probabilities[0][idx] * 100 for idx in top_n_indices]
    return top_n_genres, prediction_percentages


  