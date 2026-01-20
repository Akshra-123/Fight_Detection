from deepface import DeepFace
import cv2
import os

# ------------ CONFIG ------------
DB_PATH = "Dataset"
THRESHOLD = 0.6

VIDEO_PATH = r"C:\Users\akshr\OneDrive\Desktop\Hostel_Fight_Project\Videos\Fight_Video.mp4"
OUTPUT_DIR = r"C:\Users\akshr\OneDrive\Desktop\Hostel_Fight_Project\Output_Frames"
# --------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Video open nahi ho rahi")
    exit()

frame_id = 0
print("▶️ Real-time processing started (Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # 🔑 IMPORTANT: copy frame for annotation + saving
    output_frame = frame.copy()

    try:
        faces = DeepFace.extract_faces(
            img_path=output_frame,
            detector_backend="opencv",
            enforce_detection=False
        )
    except:
        faces = []

    # ----- NO FACE CASE -----
    if len(faces) == 0:
        cv2.putText(
            output_frame,
            "NO FACE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ----- FACE FOUND -----
    for face in faces:
        area = face["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]

        face_img = output_frame[y:y+h, x:x+w]
        label = "UNKNOWN"

        try:
            result = DeepFace.find(
                img_path=face_img,
                db_path=DB_PATH,
                model_name="VGG-Face",
                enforce_detection=False
            )

            if len(result) > 0 and len(result[0]) > 0:
                best = result[0].iloc[0]
                if best["distance"] < THRESHOLD:
                    label = os.path.basename(
                        os.path.dirname(best["identity"])
                    )
        except:
            pass

        # draw bounding box + label
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            output_frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # 🔴 REAL-TIME DISPLAY
    cv2.imshow("Hostel Face Recognition (Real-Time)", output_frame)

    # 🟢 SAVE SAME FRAME
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"frame_{frame_id:04d}.jpg"),
        output_frame
    )

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ DONE")
print(f"📁 Frames saved in: {OUTPUT_DIR}")
