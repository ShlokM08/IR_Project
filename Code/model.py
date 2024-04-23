import logging
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ExifTags
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging 
logging.basicConfig(level=logging.INFO)



app = Flask(__name__, static_folder='frontend')
CORS(app)  # Ensure CORS is enabled

model_path = 'frontend/model_file/model_optimal.h5'
model = load_model(model_path)
weights_path = 'frontend/model_file/model_weights.weights.h5'
model.load_weights(weights_path)

#Action Model
action_model_path = 'frontend/model_file/my_model_action_recognition.h5'
action_model = load_model(action_model_path)
action_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Energy levels for actions based on the attached graphs
action_energy_levels = {
    'high_intensity': (0.7, 1.0),  # High energy level range
    'medium_intensity': (0.4, 0.7),  # Medium energy level range
    'low_intensity': (0, 0.4)  # Low energy level range
}



client_id = '1a955744b8014ad6b1e80a0d7501b081'
client_secret = '4e028e2a35af49daaf5625f532481b08'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


playlist_ids = {
    'english': {
        'Positive': 'https://open.spotify.com/playlist/37i9dQZF1DX9XIFQuFvzM4?si=0be74a602a4f4614',
        'Neutral': 'https://open.spotify.com/playlist/37i9dQZF1EIWfUH5fvLgHm?si=4d27668ae34241ac',
        'Negative': 'https://open.spotify.com/playlist/37i9dQZF1DWSqBruwoIXkA?si=4d19eedbeb33434b',
        'disgust': 'https://open.spotify.com/playlist/37i9dQZF1EIUDtahSUEMYS?si=6c06a6505c8e4d62',
        'neutral': 'https://open.spotify.com/playlist/37i9dQZF1EIVVAaLvNJxGC?si=89e460a3337a47cb',
        'sad':'https://open.spotify.com/playlist/37i9dQZF1DWSqBruwoIXkA?si=4d19eedbeb33434b',
        'happy':'https://open.spotify.com/playlist/37i9dQZF1DX9XIFQuFvzM4?si=0be74a602a4f4614',
        'surprised': 'https://open.spotify.com/playlist/37i9dQZF1DX9XIFQuFvzM4?si=0be74a602a4f4614',
        'anger': 'https://open.spotify.com/playlist/37i9dQZF1EIWfUH5fvLgHm?si=4d27668ae34241ac',
        'fear' : 'https://open.spotify.com/playlist/0J2Io2I47rfYsdir02OA3i?si=0233a9e48c5043d5'

        
    },
    'hindi': {
        'Positive': 'https://open.spotify.com/playlist/2QnLDxeZMzIoCno54I9vKj?si=1b65c34640a341d0',
        'Neutral': 'https://open.spotify.com/playlist/0J2Io2I47rfYsdir02OA3i?si=0233a9e48c5043d5',
        'Negative': 'https://open.spotify.com/playlist/4YOfhHpjPB0tq29NPpDY3F?si=67fd17b93b614754',
        'disgust': 'https://open.spotify.com/playlist/4YOfhHpjPB0tq29NPpDY3F?si=67fd17b93b614754',
        'neutral': 'https://open.spotify.com/playlist/0J2Io2I47rfYsdir02OA3i?si=0233a9e48c5043d5',
        'sad':'https://open.spotify.com/playlist/4YOfhHpjPB0tq29NPpDY3F?si=67fd17b93b614754',
        'happy' :  'https://open.spotify.com/playlist/2QnLDxeZMzIoCno54I9vKj?si=1b65c34640a341d0',
        'anger': 'https://open.spotify.com/playlist/0J2Io2I47rfYsdir02OA3i?si=0233a9e48c5043d5',
        'fear' : 'https://open.spotify.com/playlist/0J2Io2I47rfYsdir02OA3i?si=0233a9e48c5043d5',
        'surprised' : 'https://open.spotify.com/playlist/2QnLDxeZMzIoCno54I9vKj?si=1b65c34640a341d0'

    },
    'tamil': {
        'Positive': 'https://open.spotify.com/playlist/2rDck89vUaM7SoGih1gzsU?si=4894c1c9791f45b1',
        'Neutral': 'https://open.spotify.com/playlist/0TOng1FTBaa6bHJcfack1S?si=9194123add144733',
        'Negative': 'https://open.spotify.com/playlist/51hmVNeZVcZQKaSecPoLs0?si=d4ad46424e194019',
        'disgust': 'https://open.spotify.com/playlist/51hmVNeZVcZQKaSecPoLs0?si=d4ad46424e194019',
        'neutral': 'https://open.spotify.com/playlist/0TOng1FTBaa6bHJcfack1S?si=9194123add144733',
        'sad':'https://open.spotify.com/playlist/51hmVNeZVcZQKaSecPoLs0?si=d4ad46424e194019',
        'happy' :  'https://open.spotify.com/playlist/2rDck89vUaM7SoGih1gzsU?si=4894c1c9791f45b1',
        'anger': 'https://open.spotify.com/playlist/0TOng1FTBaa6bHJcfack1S?si=9194123add144733',
        'fear' : 'https://open.spotify.com/playlist/0TOng1FTBaa6bHJcfack1S?si=9194123add144733',
        'surprised' : 'https://open.spotify.com/playlist/2rDck89vUaM7SoGih1gzsU?si=4894c1c9791f45b1'
        
    }
}
class_names = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 'fighting', 'hugging', 'laughing', 'listening_to_music', 'running', 'sitting', 'sleeping', 'texting', 'using_laptop']


# Emotion to category mapping
emotion_to_category = {
    'happy': 'Positive',
    'surprised': 'Positive',
    'neutral': 'neutral',
    'anger': 'Neutral',
    'fear': 'Neutral',
    'sad': 'Negative',
    'disgust': 'disgust'
}

# Valence score ranges and sort orders for each tag
valence_rules = {
    'happy': (0.66, 1, False),
    'surprised': (0.5, 0.7, True),
    'sad': (0, 0.27, True),
    'disgust': (0.2, 0.4, True),
    'neutral': (0, 0.4, False),
    'anger': (0.4, 0.7, False),
    'fear': (0.2, 0.4, True)
}

from flask import Flask, send_from_directory
import os

app = Flask(__name__, static_url_path='', static_folder='frontend')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# The rest of your Flask app code...
@app.route('/predict', methods=['POST'])
def predict():
    choice = request.form.get('choice', 'emotions')  # Default to 'emotions' if not specified

    if 'imageFile' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['imageFile']
    language = request.form.get('language', 'English')  # Default to English if not specified

    if file:
        if choice == "emotions":
            return predict_emotions(file, language)
        elif choice == "actions":
            return predict_actions(file)
    return jsonify({'error': 'Invalid request'}), 400

def predict_emotions(file, language):
    try:
        img = Image.open(file.stream)
        img = correct_image_orientation(img)
        img = process_image(img)
        emotion = predict_emotion_from_image(img)
        top_songs = get_songs_for_emotion(emotion.lower(), language)  # Pass the language to the function
        logging.info(f'Predicted Emotion: {emotion}')
        return jsonify(emotion=emotion, songs=top_songs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def predict_actions(file):
    try:
        # Convert the FileStorage object to a BytesIO object
        import io
        file_stream = io.BytesIO()
        file.save(file_stream)  # Save the file to the stream
        file_stream.seek(0)  # Go to the start of the stream
        
        logging.info("Starting to process the action prediction.")
        img = load_img(file_stream, target_size=(160, 160), color_mode='rgb')
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        logging.info("Image loaded and processed, making prediction.")
        prediction = action_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_action = class_names[predicted_class[0]]
        
        logging.info(f"Action prediction successful: {predicted_action}")
    
         # Determine the language choice
        language = request.form.get('language', 'English')
        
        # Fetch songs based on the action detected
        top_songs = get_songs_for_action(predicted_action, language)
        
        return jsonify({'action': predicted_action, 'songs': top_songs})
    except Exception as e:
        logging.error(f"Error in predict_actions: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_songs_for_action(action_tag, language='English', k=10):
    language = language.lower()
    action_tag = action_tag.lower()
    

    intensity_playlist_urls = {
        'high_intensity': {
            'english': 'https://open.spotify.com/playlist/37i9dQZF1EIfSXQEsaH9aJ?si=984072576bee4694',
            'hindi': 'https://open.spotify.com/playlist/37i9dQZF1DX8xfQRRX1PDm?si=3eeaf9ca1f95419d',
            'tamil': 'https://open.spotify.com/playlist/5AqGfimYXeEtIBX6Y5N5ja?si=e1bf98cfebd84bdf'
        },
        'medium_intensity': {
            'english': 'https://open.spotify.com/playlist/37i9dQZF1EIgk8T1xOAmM6?si=fe937e4d89384957',
            'hindi': 'https://open.spotify.com/playlist/37i9dQZF1EIcv6CMutv3XL?si=78c780d10bd643b6',
            'tamil': 'https://open.spotify.com/playlist/37i9dQZF1DXaVmfUr97Uve?si=5d88dd58f87f4371'
        },
        'low_intensity': {
            'english': 'https://open.spotify.com/playlist/37i9dQZF1DWWQRwui0ExPn?si=81abc7685e864a2a',
            'hindi': 'https://open.spotify.com/playlist/37i9dQZF1DX5q67ZpWyRrZ?si=3a2ce4f5c2504f9f',
            'tamil': 'https://open.spotify.com/playlist/1kToetbuGzM80xW1a7EoCQ?si=315eeab25ca44a17'
        }
    }
    
    # Determine intensity based on the action
    if action_tag in ['cycling', 'dancing', 'fighting', 'running']:
        action_intensity = 'high_intensity'
    elif action_tag in ['clapping', 'hugging', 'laughing']:
        action_intensity = 'medium_intensity'
    else:  # For low intensity actions
        action_intensity = 'low_intensity'

    # Determine the playlist URL based on action intensity and language
    playlist_url = intensity_playlist_urls[action_intensity][language]
    playlist_id = playlist_url.split('/')[-1].split('?')[0]

    songs_df = call_playlist("spotify",playlist_id)
    if songs_df.empty:
        return []

    # Filter songs based on energy levels
    min_energy, max_energy = action_energy_levels[action_intensity]
    filtered_songs = songs_df[(songs_df['energy'] >= min_energy) & (songs_df['energy'] <= max_energy)]

    # Conditional sampling
    if not filtered_songs.empty:
        sample_size = min(len(filtered_songs), k)  # Take the smaller of the two values
        if sample_size == 0:
            return []  # Return an empty list if there are no songs to sample
        sorted_songs = filtered_songs.sample(n=sample_size)
        songs_list = sorted_songs[['artist', 'track_name', 'energy']].to_dict(orient='records')
        return songs_list
    else:
        # Handle the case where no songs match the filter
        logging.info(f"No songs match the energy criteria for {action_tag}")
        return []

def correct_image_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        logging.warning("No EXIF orientation data found.")
    return img

def process_image(img):
    img = img.convert('L')
    img = img.resize((48, 48))
    return np.array(img) / 255.0

def predict_emotion_from_image(img_array):
    img_array = img_array.reshape(1, 48, 48, 1)  # Add batch dimension
    result = model.predict(img_array)
    predicted_class_index = np.argmax(result, axis=1)
    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    return label_dict[predicted_class_index[0]]

import logging

def get_songs_for_emotion(emotion_tag, language='English', k=10):
    language = language.lower()
    emotion_tag = emotion_tag.lower()

    logging.info(f"Fetching songs for emotion: {emotion_tag} in language: {language}")
    logging.info(f"Received language: {language}, emotion: {emotion_tag}")
    logging.info(f"Available languages: {list(playlist_ids.keys())}")
    logging.info(f"Available emotions for language: {list(playlist_ids[language].keys()) if language in playlist_ids else 'Language not found'}")

    if language in playlist_ids and emotion_tag in playlist_ids[language]:
        playlist_url = playlist_ids[language][emotion_tag]
        playlist_id = playlist_url.split('/')[-1].split('?')[0]

        logging.info(f"Using playlist URL: {playlist_url}")

        songs_df = call_playlist('spotify', playlist_id)
        if songs_df.empty:
            logging.warning("No songs fetched from Spotify.")
            return []

        min_valence, max_valence, ascending = valence_rules.get(emotion_tag, (0, 0.5, False))
        filtered_songs = songs_df[(songs_df['valence'] >= min_valence) & (songs_df['valence'] <= max_valence)]
        sorted_songs = filtered_songs.sort_values(by='valence', ascending=ascending).head(k)
        songs_list = sorted_songs[['artist', 'track_name', 'valence']].to_dict(orient='records')

        logging.info(f"Returning {len(songs_list)} songs.")
        return songs_list
    else:
        logging.error("Invalid language or emotion tag.")
        return []

def call_playlist(creator, playlist_id):
      #step1

    playlist_features_list = ["artist","album","track_name",  "track_id","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]

    playlist_df = pd.DataFrame(columns = playlist_features_list)

        #step2

    playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    for track in playlist:
        # Create empty dict
        playlist_features = {}
        # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]

        # Get audio features
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[4:]:
            playlist_features[feature] = audio_features[feature]
        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)

    #Step 3
    return playlist_df

if __name__ == '__main__':
    app.run(debug=True)
