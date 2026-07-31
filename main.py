import cv2


DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

REFERENCE_IMAGE = "images/picture3.jpeg"

THRESHOLD = 1.128


detector = cv2.FaceDetectorYN_create(
    DETECTOR_MODEL,
    "",
    (320, 320)
)

recognizer = cv2.FaceRecognizerSF_create(
    RECOGNIZER_MODEL,
    ""
)


def get_face_embedding(image):
    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        return None, None

    face = faces[0]

    aligned_face = recognizer.alignCrop(image, face)

    embedding = recognizer.feature(aligned_face)

    return embedding, face


reference_image = cv2.imread(REFERENCE_IMAGE)

if reference_image is None:
    raise ValueError("Could not load reference image")

reference_embedding, _ = get_face_embedding(reference_image)

if reference_embedding is None:
    raise ValueError("No face detected in reference image")


camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    embedding, face = get_face_embedding(frame)

    if embedding is not None:
        distance = recognizer.match(
            reference_embedding,
            embedding,
            cv2.FaceRecognizerSF_FR_NORM_L2
        )

        if distance <= THRESHOLD:
            result = "Same person"
        else:
            result = "Different person"

        x, y, width, height = face[:4].astype(int)

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{result} | Distance: {distance:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()