import cv2

from database import load_embeddings

from recognition import (
    detector,
    recognizer,
    identify_face
)

from registration import import_known_faces_to_db


# Import new known faces into PostgreSQL
import_known_faces_to_db()


# Load known embeddings from PostgreSQL
known_faces = load_embeddings()

print(
    f"Loaded {len(known_faces)} "
    f"embeddings from PostgreSQL"
)


# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


while True:
    success, frame = camera.read()

    if not success:
        break

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            aligned_face = recognizer.alignCrop(
                frame,
                face
            )

            current_embedding = recognizer.feature(
                aligned_face
            )

            best_name, best_distance = identify_face(
                current_embedding,
                known_faces
            )

            x, y, face_width, face_height = (
                face[:4].astype(int)
            )

            cv2.rectangle(
                frame,
                (x, y),
                (
                    x + face_width,
                    y + face_height
                ),
                (0, 200, 0),
                2
            )

            cv2.putText(
                frame,
                (
                    f"{best_name} | "
                    f"Distance: {best_distance:.2f}"
                ),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "Face Identification",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()