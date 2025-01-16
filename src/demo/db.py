import json
import sqlite3

import numpy as np
from iris import IrisTemplate
from iris.nodes.matcher.utils import simple_hamming_distance
from sklearn.metrics.pairwise import cosine_similarity

DATABASE_NAME = "iris_app.db"

OPEN_IRIS_TRASHHOLD = 0.375
FULL_EYE_TRASHHOLD = 0.06
NORMALIZED_IRIS_TRASHHOLD = 0.06


def initialize_database():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE
        )
    """)

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
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    is_new_user = False

    try:
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        user_id = cursor.lastrowid
        is_new_user = True
    except sqlite3.IntegrityError:
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()[0]
    finally:
        cursor.execute("""
            INSERT INTO feature_vectors (userId, classic_embedding, resnet_embedding, resnet_normalized_embedding)
            VALUES (?, ?, ?, ?)
        """, (user_id, classic_embedding, resnet_embedding, resnet_normalized_embedding))

        conn.commit()
        conn.close()
        return is_new_user


def fetch_all_feature_vectors():
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
    embedding_list_gallery = np.array(embedding_list_gallery)
    embedding_list_probe = np.array(embedding_list_probe).reshape(1, -1)

    similarity = -cosine_similarity(embedding_list_probe, embedding_list_gallery)
    distance = (similarity + 1) / 2
    return distance


def find_all_matches_under_threshold(classic_embedding, resnet_embedding, resnet_normalized_embedding):
    deserialized_classic_embedding = IrisTemplate.deserialize(classic_embedding)

    resnet_embedding = np.array(resnet_embedding)
    resnet_normalized_embedding = np.array(resnet_normalized_embedding)

    records = fetch_all_feature_vectors()

    classic_embedding_gallery = []
    resnet_embedding_gallery = []
    resnet_normalized_embedding_gallery = []
    user_ids = []

    for record in records:
        user_id, classic_stored, resnet_stored, resnet_normalized_stored = record

        classic_embedding_gallery.append(IrisTemplate.deserialize(json.loads(classic_stored)))

        resnet_embedding_gallery.extend(json.loads(resnet_stored))
        resnet_normalized_embedding_gallery.extend(json.loads(resnet_normalized_stored))

        user_ids.append(user_id)

    resnet_embedding_gallery_array = np.array(resnet_embedding_gallery)
    resnet_normalized_embedding_gallery_array = np.array(
        resnet_normalized_embedding_gallery)

    resnet_distance = compute_cosine_distance(resnet_embedding_gallery_array, resnet_embedding)
    resnet_normalized_distance = compute_cosine_distance(resnet_normalized_embedding_gallery_array,
                                                         resnet_normalized_embedding)

    matches_under_threshold = []

    for i in range(len(user_ids)):
        classic_distance = simple_hamming_distance(deserialized_classic_embedding, classic_embedding_gallery[i])[0]

        resnet_distance_i = resnet_distance[0][i]

        resnet_normalized_distance_i = resnet_normalized_distance[0][i]

        if classic_distance < OPEN_IRIS_TRASHHOLD:
            matches_under_threshold.append(("OpenIrisLibrary", classic_distance, user_ids[i]))
        if resnet_distance_i < FULL_EYE_TRASHHOLD:
            matches_under_threshold.append(("Full Eye ResNet", resnet_distance_i, user_ids[i]))
        if resnet_normalized_distance_i < NORMALIZED_IRIS_TRASHHOLD:
            matches_under_threshold.append(("Normalized Eye ResNet", resnet_normalized_distance_i, user_ids[i]))

    if not matches_under_threshold:
        return [], False

    matches_under_threshold.sort(key=lambda x: x[1])
    final_matches = []
    for match in matches_under_threshold:
        method, distance, user_id = match

        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        username = cursor.fetchone()[0]
        conn.close()

        final_matches.append({"method": method, "distance": distance, "label": username})

    return final_matches, True
