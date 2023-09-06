import streamlit as st
import tempfile
import os
import shutil
import datetime
import random
from pydub import AudioSegment
from component.pred import predict_top_genres, predict_ML, genres
from component.convert import convert_mp3_to_wav

# Capture l'erreur si le package dotenv n'est pas installé.
# Si non execute le code 
try:
    import dotenv
except ImportError:
    print("Le package dotenv n'est pas installé.")
else:
    # Votre code qui utilise le package dotenv ici
    # Par exemple, charger les variables d'environnement à partir d'un fichier .env
    dotenv.load_dotenv()


def create_temp_directory():
    return tempfile.TemporaryDirectory()

def save_uploaded_file(uploaded_file, temp_dir):
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    
    if uploaded_file.name.endswith(".mp3"):
        convert_mp3_to_wav(uploaded_file, temp_file_path)

    elif uploaded_file.name.endswith(".wav"):
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_file_path

def generate_random_extracts(audio, num_extracts):
    max_extract_duration_ms = 30 * 1000
    audio_duration_ms = len(audio)
    extract_file_paths = []

    for _ in range(num_extracts):
        extract_start_ms = random.randint(
            10 * 1000, audio_duration_ms - max_extract_duration_ms - 10 * 1000
        )
        extract_end_ms = extract_start_ms + max_extract_duration_ms
        extract = audio[extract_start_ms:extract_end_ms]

        extract_path = os.path.join(temp_dir.name, f"extract_{len(extract_file_paths)}.wav")
        extract.export(extract_path, format="wav")
        extract_file_paths.append(extract_path)

    return extract_file_paths

def process_and_display_results(extract_paths):
    top_genres_lists = []
    prediction_percentages_lists = []
    top_genres_azure = []

    for extract_path in extract_paths:
        with st.spinner("Classification en cours ..."):
            top_genres, prediction_percentages = predict_top_genres(extract_path, top_n=3)
            result_autoML = predict_ML(extract_path)
        
        top_genres_lists.append(top_genres)
        prediction_percentages_lists.append(prediction_percentages)
        top_genres_azure.append(result_autoML)

        # st.audio(extract_path, format='audio/wav')
    most_frequent_class = max(set(top_genres_azure), key=top_genres_azure.count)
    
 
    for i, (top_genres, prediction_percentages) in enumerate(zip(top_genres_lists, prediction_percentages_lists)):
        formatted_genres = [f"{genre} {percentage:.2f}%" for genre, percentage in zip(top_genres, prediction_percentages)]
        print(f"Extrait {i + 1}: {', '.join(formatted_genres)}")

    print(top_genres_azure)
    
    genre_avg_percentages = {}
    for genres, percentages in zip(top_genres_lists, prediction_percentages_lists):
        for genre, percentage in zip(genres, percentages):
            if genre not in genre_avg_percentages:
                genre_avg_percentages[genre] = 0
            genre_avg_percentages[genre] += percentage

    for genre in genre_avg_percentages:
        genre_avg_percentages[genre] /= len(top_genres_lists)

    
    sorted_genres = sorted(genre_avg_percentages.keys(), key=lambda x: -genre_avg_percentages[x])

    
    st.subheader("Résultat:")
    table_data = []
    for i, genre in enumerate(sorted_genres[:3]):
        table_data.append({
            "Genre": f"{genre} {genre_avg_percentages[genre]:.2f}%","Azure autoML": ""})

    table_data[0]["Azure autoML"] = most_frequent_class

    st.table(table_data)



# Titre de l'application
st.title("Identification de genres musicaux")

upload_folder = "audio_files"
os.makedirs(upload_folder, exist_ok=True)

uploaded_file = st.file_uploader("Uploader un extrait audio", type=["wav", "mp3"])

if uploaded_file is not None:
    temp_dir = create_temp_directory()
    temp_file_path = save_uploaded_file(uploaded_file, temp_dir)
    st.audio(temp_file_path, format='audio/wav')

    audio = AudioSegment.from_file(temp_file_path)
    audio_duration_ms = len(audio)

    if audio_duration_ms > 30 * 1000:
        num_random_extracts = 5 
        extract_paths = generate_random_extracts(audio, num_random_extracts)
        process_and_display_results(extract_paths)
    else:
        process_and_display_results([temp_file_path])

    st.subheader("Corriger le genre:")
    with st.form("Genre"):
        selected_genre = st.selectbox("Genre:", genres)
        submitted = st.form_submit_button("Valider")

    if submitted:
        st.write(f"genre: {selected_genre}")

        if uploaded_file:
            uploaded_file_name = uploaded_file.name
            uploaded_file_path = os.path.join(upload_folder, uploaded_file_name)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            uploaded_file_name = f"{uploaded_file_name}_{timestamp}.wav"
            uploaded_file_path = os.path.join(upload_folder, uploaded_file_name)
            shutil.copy(temp_file_path, uploaded_file_path)
