# import required library
import cv2
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

path_weights = "best_model_weights.pth"
model = mobilenet_v3_large(weights=None, num_classes=7)
model.load_state_dict(torch.load(path_weights))

# Define the emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define the image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
cap = cv2.VideoCapture(0)
# Read frames from the webcam until the user presses 'q'
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("Failed to read frame from the webcam")
        break

    # Perform face detection on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)

    # Loop over all the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y - 20), (x + w + 20, y + h + 30), (0, 255, 255), 2)

        # Crop the region of interest (ROI) from the original frame
        roi = frame[y - 20:y + h + 30, x:x + w]

        # Preprocess the ROI image
        input_image = transform(roi).unsqueeze(0)

        # Make predictions using the pre-trained model
        output = torch.nn.functional.softmax(model(input_image), dim=-1)

        # Get the predicted emotion label
        prob, predicted_idx = torch.max(output, dim=1)
        predicted_emotion = emotion_labels[predicted_idx.item()]

        # Display the predicted emotion label on the frame
        cv2.putText(frame, predicted_emotion + " " + str(round(prob.item(), 2)), (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Display the frame in a window
    cv2.imshow('Face Detection', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()