import psycopg
import numpy as np


DB_CONFIG = {
    "dbname": "face_recognition",
    "user": "face_user",
    "password": "face_password",
    "host": "localhost",
    "port": 5432
}


def get_connection():
    return psycopg.connect(**DB_CONFIG)


def add_person(name):
    connection = get_connection()

    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO persons (name) VALUES (%s) RETURNING id;",
            (name,)
        )

        person_id = cursor.fetchone()[0]

    connection.commit()
    connection.close()

    return person_id


def get_all_persons():
    connection = get_connection()

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, name, created_at FROM persons ORDER BY id;"
        )

        persons = cursor.fetchall()

    connection.close()

    return persons


def save_embedding(person_id, image_path, embedding):
    connection = get_connection()

    embedding_bytes = embedding.astype(np.float32).tobytes()

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO face_embeddings
                (person_id, image_path, embedding)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (person_id, image_path, embedding_bytes)
        )

        embedding_id = cursor.fetchone()[0]

    connection.commit()
    connection.close()

    return embedding_id


def load_embeddings():
    connection = get_connection()

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                fe.id,
                fe.person_id,
                p.name,
                fe.image_path,
                fe.embedding
            FROM face_embeddings AS fe
            JOIN persons AS p
                ON fe.person_id = p.id
            ORDER BY fe.id;
            """
        )

        rows = cursor.fetchall()

    connection.close()

    embeddings = []

    for row in rows:
        embedding_id = row[0]
        person_id = row[1]
        person_name = row[2]
        image_path = row[3]
        embedding_bytes = row[4]

        embedding = np.frombuffer(
            embedding_bytes,
            dtype=np.float32
        ).reshape(1, -1)

        embeddings.append(
            {
                "id": embedding_id,
                "person_id": person_id,
                "name": person_name,
                "image_path": image_path,
                "embedding": embedding
            }
        )

    return embeddings