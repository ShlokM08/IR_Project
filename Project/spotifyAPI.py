import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import requests
from requests.exceptions import RetryError
from spotipy import SpotifyException

# Spotify API credentials
cid = '818b87c2df5a45f48d3705e8b930edd1'
secret = '25273da78dc24e98833288ba169f9691'

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def call_api_with_retry(spotify_function, *args, **kwargs):
    max_retries = 5
    retry_count = 0
    backoff_factor = 2
    while retry_count < max_retries:
        try:
            return spotify_function(*args, **kwargs)
        except SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers.get('Retry-After', 1))
                sleep_time = retry_after * (backoff_factor ** retry_count)
                time.sleep(sleep_time)
                retry_count += 1
            else:
                raise
    raise RetryError(f"Max retries exceeded with url: {kwargs.get('url')}")


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

# Add the actual playlist IDs for each category
playlist_ids = {
    'Positive': 'https://open.spotify.com/playlist/37i9dQZF1DX9XIFQuFvzM4?si=0be74a602a4f4614',
    'Neutral': 'https://open.spotify.com/playlist/37i9dQZF1EIWfUH5fvLgHm?si=4d27668ae34241ac',
    'Negative': 'https://open.spotify.com/playlist/37i9dQZF1DWSqBruwoIXkA?si=4d19eedbeb33434b',
    'disgust': 'https://open.spotify.com/playlist/37i9dQZF1EIUDtahSUEMYS?si=6c06a6505c8e4d62',
    'neutral': 'https://open.spotify.com/playlist/37i9dQZF1EIVVAaLvNJxGC?si=89e460a3337a47cb',

}

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

def get_songs_for_emotion(emotion_tag, k=10):
    # Determine the playlist based on emotion category
    category = emotion_to_category[emotion_tag]
    playlist_id = playlist_ids[category]

    # Fetch songs from the playlist
    songs_df = call_playlist('spotify', playlist_id)

    # Apply valence rules for sorting
    min_valence, max_valence, ascending = valence_rules[emotion_tag]
    filtered_songs = songs_df[(songs_df['valence'] >= min_valence) & (songs_df['valence'] <= max_valence)]
    sorted_songs = filtered_songs.sort_values(by='valence', ascending=ascending).head(k)

    return sorted_songs[['artist', 'track_name', 'valence']]

# Example usage
# Example usage
emotion_tag = input("Enter the emotion tag: ")
top_songs = get_songs_for_emotion(emotion_tag)

# Printing results in an aligned format
print(top_songs.to_string(index=False))
