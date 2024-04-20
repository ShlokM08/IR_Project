import spotipy
pip install spotipy

from spotipy.oauth2 import SpotifyClientCredentials

import pandas as pd


cid = '818b87c2df5a45f48d3705e8b930edd1'
secret = '25273da78dc24e98833288ba169f9691'

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)

sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

#

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

call_playlist('spotify','https://open.spotify.com/playlist/37i9dQZF1EIhmSBwUDxg84?si=e431748c16ec42e4')