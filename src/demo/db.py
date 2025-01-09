import json
import sqlite3

import numpy as np
from iris import IrisTemplate
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation.hamming_distance_cuda import simple_hamming_distance

DATABASE_NAME = "iris_app.db"


def initialize_database():
    """Initialize the SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE
        )
    """)

    # Create feature_vectors table with separate columns for each embedding type
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_vectors (
            userId INTEGER NOT NULL,
            classic_embedding TEXT NOT NULL,
            resnet_embedding TEXT NOT NULL,
            resnet_normalized_embedding TEXT NOT NULL,
            FOREIGN KEY (userId) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def insert_user(username, classic_embedding, resnet_embedding, resnet_normalized_embedding):
    """Insert a new user and their feature vectors into the database."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    is_new_user = False

    try:
        # Insert username into users table
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        user_id = cursor.lastrowid  # Get the ID of the newly inserted user
        is_new_user = True
    except sqlite3.IntegrityError:
        # If the user already exists, fetch their ID
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()[0]
    finally:
        # Insert feature vectors into feature_vectors table
        cursor.execute("""
            INSERT INTO feature_vectors (userId, classic_embedding, resnet_embedding, resnet_normalized_embedding)
            VALUES (?, ?, ?, ?)
        """, (user_id, classic_embedding, resnet_embedding, resnet_normalized_embedding))

        conn.commit()
        conn.close()
        return is_new_user


def fetch_all_feature_vectors():
    """Fetch all feature vectors from the database."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT userId, classic_embedding, resnet_embedding, resnet_normalized_embedding
        FROM feature_vectors
    """)
    records = cursor.fetchall()
    conn.close()
    return records


def compute_cosine_distance(embedding_list_gallery, embedding_list_probe):
    """Compute cosine distance between a gallery of embeddings and a probe embedding."""
    # Ensure the embeddings are 2D arrays
    embedding_list_gallery = np.array(embedding_list_gallery)  # Shape: (n_samples, n_features)
    embedding_list_probe = np.array(embedding_list_probe).reshape(1, -1)  # Shape: (1, n_features)

    # Compute cosine similarity
    similarity = -cosine_similarity(embedding_list_probe, embedding_list_gallery)
    distance = (similarity + 1) / 2  # Normalize to [0, 1]
    return distance


def find_top_3_matches(classic_embedding, resnet_embedding, resnet_normalized_embedding):
    """Find the top 3 closest matches in the database for the given embeddings."""
    # Deserialize the classic embedding
    deserialized_classic_embedding = IrisTemplate.deserialize(classic_embedding)

    # Convert resnet embeddings to numpy arrays
    resnet_embedding = np.array(resnet_embedding)
    resnet_normalized_embedding = np.array(resnet_normalized_embedding)

    # Fetch all records from the database
    records = fetch_all_feature_vectors()

    # Lists to store gallery embeddings and user IDs
    classic_embedding_gallery = []
    resnet_embedding_gallery = []
    resnet_normalized_embedding_gallery = []
    user_ids = []

    # Populate the gallery lists
    for record in records:
        user_id, classic_stored, resnet_stored, resnet_normalized_stored = record

        # Deserialize and store classic embeddings
        classic_embedding_gallery.append(IrisTemplate.deserialize(json.loads(classic_stored)))

        # Deserialize and store resnet embeddings
        resnet_embedding_gallery.extend(json.loads(resnet_stored))
        resnet_normalized_embedding_gallery.extend(json.loads(resnet_normalized_stored))

        # Store user IDs
        user_ids.append(user_id)

    # Convert resnet embeddings to numpy arrays
    resnet_embedding_gallery_array = np.array(resnet_embedding_gallery)  # Shape: (n_samples, n_features)
    resnet_normalized_embedding_gallery_array = np.array(
        resnet_normalized_embedding_gallery)  # Shape: (n_samples, n_features)

    # Compute cosine distances
    resnet_distance = compute_cosine_distance(resnet_embedding_gallery_array, resnet_embedding)
    resnet_normalized_distance = compute_cosine_distance(resnet_normalized_embedding_gallery_array,
                                                         resnet_normalized_embedding)

    # List to store all matches
    all_matches = []

    # Compute distances for each method
    for i in range(len(user_ids)):
        # Classic embedding distance
        classic_distance = simple_hamming_distance(deserialized_classic_embedding, classic_embedding_gallery[i])[0]

        # Resnet embedding distance
        resnet_distance_i = resnet_distance[0][i]

        # Normalized resnet embedding distance
        resnet_normalized_distance_i = resnet_normalized_distance[0][i]

        # Add all matches to the list
        all_matches.append(("classic", classic_distance, user_ids[i]))
        all_matches.append(("resnet_images", resnet_distance_i, user_ids[i]))
        all_matches.append(("resnet_normalized", resnet_normalized_distance_i, user_ids[i]))

    # Sort all matches by distance
    all_matches.sort(key=lambda x: x[1])

    # Fetch the top 3 matches
    top_3_matches = []
    # for match in all_matches[:3]:
    for match in all_matches:
        method, distance, user_id = match

        # Fetch the username for the user
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        username = cursor.fetchone()[0]
        conn.close()

        top_3_matches.append({"method": method, "distance": distance, "label": username})

    return top_3_matches
