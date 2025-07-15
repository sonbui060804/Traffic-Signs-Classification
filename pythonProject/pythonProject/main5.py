import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục chứa dữ liệu
DATA_PATH = 'myData'  # Thay bằng đường dẫn đến tập dữ liệu
IMG_SIZE = 32  # Kích thước ảnh đầu vào
NUM_CLASSES = 43  # Số lượng lớp (biển báo) trong GTSRB

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Chuẩn hóa
    return img

# Tải dữ liệu
def load_data(data_dir):
    images = []
    labels = []
    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(data_dir, str(class_id))
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(class_id)
    return np.array(images), np.array(labels)

# Xây dựng mô hình CNN
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Tải và chia dữ liệu
images, labels = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = build_model()
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Lưu mô hình
model.save('traffic_sign_model.h5')

# Dự đoán trên một ảnh mới
def predict_image(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    return predicted_class

# Ví dụ sử dụng
# predicted_class = predict_image('path_to_new_image.jpg')
# print(f"Predicted traffic sign class: {predicted_class}")