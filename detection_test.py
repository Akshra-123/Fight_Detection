import cv2

VIDEO_PATH = r"C:\Users\akshr\OneDrive\Desktop\Hostel_Fight_Project\Videos\Fight_Video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Video open nahi hui")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ALWAYS DRAW SOMETHING
    cv2.putText(
        frame,
        "VIDEO RUNNING",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("DEBUG VIDEO", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
