from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
from mutagen import File
from mutagen.id3 import APIC
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


CAMELOT_MAP = {
    'C': '8B', 'C#': '3B', 'D': '10B', 'D#': '5B', 'E': '12B',
    'F': '7B', 'F#': '2B', 'G': '9B', 'G#': '4B', 'A': '11B', 'A#': '6B', 'B': '1B'
}

def camelot_neighbors(camelot_key):
    num = int(camelot_key[:-1])
    mode = camelot_key[-1]
    neighbors = [
        f"{num}{mode}",                         # Same key
        f"{(num % 12) + 1}{mode}",              # +1 clockwise
        f"{(num - 2) % 12 + 1}{mode}",          # -1 counterclockwise
        f"{num}{'A' if mode == 'B' else 'B'}"   # Switch major/minor
    ]
    return neighbors


def transition_score(song1, song2): 
    # Higher weight on key compatibility
    bpm_diff = abs(song1['bpm'] - song2['bpm'])

    key1 = CAMELOT_MAP.get(song1['key'])
    key2 = CAMELOT_MAP.get(song2['key'])

    # Penalize bad key transitions
    if not key1 or not key2:
        key_penalty = 100  # unknown = heavy penalty
    elif key2 in camelot_neighbors(key1):
        key_penalty = 0  # ideal
    else:
        key_penalty = 50  # harsh but not total blocker

    # Weighted score: key_penalty dominates
    return key_penalty + bpm_diff * 0.5

def order_songs_greedy(songs):
    if not songs:
        return []
    
    songs = sorted(songs, key=lambda s: s['bpm'])  # sort by bpm to bias direction
    ordered = [songs.pop(0)]  # start with slowest track

    while songs:
        last = ordered[-1]
        # Pick the best next song based on transition score
        next_song = min(songs, key=lambda s: transition_score(last, s))
        ordered.append(next_song)
        songs.remove(next_song)

    return ordered


@app.route('/order', methods=['POST'])
def order_songs():
    try:
        songs = request.json  # expects list of song dicts
        ordered = order_songs_greedy(songs.copy())
        return jsonify(ordered)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def extract_artwork(audio_file):
    try:
        audio = File(audio_file)
        if audio is None:
            return None
        artwork = None
        if audio.tags is not None:
            for tag in audio.tags.values():
                if isinstance(tag, APIC):
                    artwork = tag.data
                    break
        if artwork:
            # Convert bytes to base64 string for frontend display
            img = Image.open(BytesIO(artwork))
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            encoded_art = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return encoded_art
    except Exception as e:
        print("Artwork extraction error:", e)
    return None

def analyze_file(file_stream):
    file_stream.seek(0)
    y, sr = librosa.load(file_stream, sr=None, mono=True, duration = 20.0)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print(type(tempo), tempo)
    tempo = float(tempo)  # convert to scalar float
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key_index = chroma_mean.argmax()
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = notes[key_index]

    return round(tempo), key



@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist('files')
    results = []

    for f in files:
        filename = f.filename

        # Read file into bytes buffer for librosa and mutagen
        file_bytes = f.read()
        file_stream = BytesIO(file_bytes)

        # Analyze BPM/key
        try:
            bpm, key = analyze_file(file_stream)
        except Exception as e:
            bpm, key = None, None
            print(f"Error analyzing {filename}: {e}")

        # Reset stream for artwork extraction
        file_stream.seek(0)
        artwork = extract_artwork(file_stream)

        results.append({
            "title": filename,
            "bpm": bpm,
            "key": key,
            "artwork": artwork  # base64 jpeg or None
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
