import os
import cv2

from database import (
    get_or_create_person,
    save_embedding,
    embedding_exists
)

from recognition import get_face_embedding


KNOWN_FACES_DIR = "known_faces"


def register_person(name, image_paths):
    person_id = get_or_create_person(name)

    saved_embeddings = 0

    for image_path in image_paths:
        if embedding_exists(image_path):
            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        embedding = get_face_embedding(image)

        if embedding is None:
            continue

        save_embedding(
            person_id=person_id,
            image_path=image_path,
            embedding=embedding
        )

        saved_embeddings += 1

    return saved_embeddings


def import_known_faces_to_db():
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(
            KNOWN_FACES_DIR,
            person_name
        )

        if not os.path.isdir(person_path):
            continue

        image_paths = []

        for image_name in os.listdir(person_path):
            image_path = os.path.join(
                person_path,
                image_name
            )

            image_paths.append(image_path)

        register_person(
            person_name,
            image_paths
        )