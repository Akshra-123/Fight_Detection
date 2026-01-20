from deepface import DeepFace
import cv2
import os
from collections import defaultdict, deque, Counter

# ================= CONFIG =================
DB_PATH = "Dataset"
VIDEO_PATH = r"C:\Users\akshr\OneDrive\Desktop\Hostel_Fight_Project\Videos\Fight_Video.mp4"

THRESHOLD = 0.85            # relaxed for video
MEMORY_SIZE = 10            # per-person voting window
DISPLAY_FRAMES = 75         # ~3 seconds lock (25 FPS)
# =========================================

# Haar cascade (stable detection)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Video open nahi hui")
    exit()

# -------- MEMORY STRUCTURES --------
person_memory = defaultdict(lambda: deque(maxlen=MEMORY_SIZE))

last_display_label = "SCANNING..."
display_counter = 0

print("▶️ Final stable detection started (Press Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for stability
    frame = cv2.resize(frame, (640, 360))
    output = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ---------------- NO FACE ----------------
    if len(faces) == 0:
        if display_counter <= 0:
            last_display_label = "NO FACE"
        display_counter = max(display_counter - 1, 0)

    # ---------------- FACE FOUND ----------------
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        label = "UNKNOWN"

        try:
            result = DeepFace.find(
                img_path=face_img,
                db_path=DB_PATH,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=False
            )

            if len(result) > 0 and len(result[0]) > 0:
                top_matches = result[0].head(3)
                for _, row in top_matches.iterrows():
                    if row["distance"] < THRESHOLD:
                        label = os.path.basename(
                            os.path.dirname(row["identity"])
                        )
                        break
        except:
            pass

        # ---- TEMPORAL VOTING (ANGLE FIX) ----
        person_id = (x // 50, y // 50)   # rough tracking
        person_memory[person_id].append(label)

        stable_label = Counter(
            person_memory[person_id]
        ).most_common(1)[0][0]

        # ---- DISPLAY LOCK (READABLE FIX) ----
        if stable_label not in ["UNKNOWN", "NO FACE"]:
            last_display_label = stable_label
            display_counter = DISPLAY_FRAMES

        # Draw face box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show per-face label
        cv2.putText(
            output,
            stable_label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # -------- GLOBAL DISPLAY (LOCKED) --------
    if display_counter > 0:
        display_counter -= 1
        show_label = last_display_label
    else:
        show_label = "SCANNING..."

    cv2.putText(
        output,
        f"IDENTIFIED: {show_label}",
        (20, output.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Hostel Fight Face Recognition - FINAL", output)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Finished successfully")
