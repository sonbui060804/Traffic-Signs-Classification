import numpy as np
import cv2
from tensorflow.keras.models import load_model

#############################################
frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
#############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRAINED MODEL
model = load_model("traffic_sign_model.h5")  # Load the .h5 file directly

# FUNCTIONS
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    # Keep the image in BGR format (3 channels) instead of converting to grayscale
    img = cv2.resize(img, (32, 32))  # Resize first
    img = img / 255.0  # Normalize
    return img  # Return shape (32, 32, 3)

def getCalssName(classNo):
    classNames = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    if 0 <= classNo < len(classNames):
        return classNames[classNo]
    else:
        return "Unknown"

# MAIN LOOP
while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Không đọc được từ camera")
        break

    # PROCESS IMAGE
    img = np.asarray(imgOriginal)
    img = preprocessing(img)

    # CHUYỂN VỀ DẠNG HIỂN THỊ ĐƯỢC (optional, for visualization)
    img_display = (img * 255).astype(np.uint8)
    cv2.imshow("Processed Image", img_display)

    # CHUẨN BỊ CHO MODEL
    img = img.reshape(1, 32, 32, 3)  # Reshape to include 3 channels

    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        cv2.putText(imgOriginal, str(classIndex) + " " + str(getCalssName(classIndex)),
                    (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, "Probability: " + str(round(probabilityValue * 100, 2)) + "%",
                    (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Output", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# RELEASE RESOURCES
cap.release()
cv2.destroyAllWindows()