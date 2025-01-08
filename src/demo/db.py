import sqlite3
import json
import random

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

def create_embeddings(image, side):
    """Generate random feature vectors and return them as JSON strings."""
    classic_embedding = [random.uniform(0, 1) for _ in range(100)]
    resnet_embedding = [random.uniform(0, 1) for _ in range(100)]
    resnet_normalized_embedding = [random.uniform(0, 1) for _ in range(100)]

    # Convert lists to JSON strings
    classic_embedding_str = json.dumps(classic_embedding)
    resnet_embedding_str = json.dumps(resnet_embedding)
    resnet_normalized_embedding_str = json.dumps(resnet_normalized_embedding)

    return classic_embedding_str, resnet_embedding_str, resnet_normalized_embedding_str

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

def hamming_distance(v1, v2):
    """Return a random value between 0 and 1."""
    return random.random()

def find_top_3_matches(classic_embedding, resnet_embedding, resnet_normalized_embedding):
    """Find the top 3 closest matches in the database for the given embeddings."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Fetch all records from the feature_vectors table
    cursor.execute("""
        SELECT userId, classic_embedding, resnet_embedding, resnet_normalized_embedding
        FROM feature_vectors
    """)
    records = cursor.fetchall()

    # List to store all matches
    all_matches = []

    for record in records:
        user_id, classic_stored, resnet_stored, resnet_normalized_stored = record

        # Compute distances for each method
        # TODO Implement the actual distance calculation of the hamming distance
        classic_distance = hamming_distance(
            classic_embedding, json.loads(classic_stored)
        )
        resnet_distance = hamming_distance(
            resnet_embedding, json.loads(resnet_stored)
        )
        resnet_normalized_distance = hamming_distance(
            resnet_normalized_embedding, json.loads(resnet_normalized_stored)
        )

        # Add all matches to the list
        all_matches.append(("classic", classic_distance, user_id))
        all_matches.append(("resnet_images", resnet_distance, user_id))
        all_matches.append(("resnet_normalized", resnet_normalized_distance, user_id))

    # Sort all matches by distance
    all_matches.sort(key=lambda x: x[1])

    # Fetch the top 3 matches
    top_3_matches = []
    for match in all_matches[:3]:
        method, distance, user_id = match

        # Fetch the username for the user
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        username = cursor.fetchone()[0]

        top_3_matches.append({
            "method": method,
            "distance": distance,
            "label": username
        })

    conn.close()
    return top_3_matches