import cv2


DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

detector = cv2.FaceDetectorYN_create(
    DETECTOR_MODEL,
    "",
    (320, 320)
)

recognizer = cv2.FaceRecognizerSF_create(
    RECOGNIZER_MODEL,
    ""
)


def get_face_embedding(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        raise ValueError(f"No face detected in: {image_path}")

    face = faces[0]

    aligned_face = recognizer.alignCrop(image, face)

    embedding = recognizer.feature(aligned_face)

    return embedding


embedding1 = get_face_embedding("images/picture1.jpeg")
embedding2 = get_face_embedding("images/picture2.jpeg")

distance = recognizer.match(
    embedding1,
    embedding2,
    cv2.FaceRecognizerSF_FR_NORM_L2
)

print("Euclidean distance:", distance)

THRESHOLD = 1.128

if distance <= THRESHOLD:
    print("Same person")
else:
    print("Different person")