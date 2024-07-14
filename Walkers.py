import cv2

# Create our body classifier
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while cap.isOpened():
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with rectangles
    cv2.imshow('Pedestrians', frame)

    # Break the loop if the space key is pressed
    if cv2.waitKey(1) == 32:
        break

cap.release()
cv2.destroyAllWindows()

