import numpy as np
import cv2
import pickle
from collections import deque, Counter

#############################################
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.9  # Ngưỡng tin cậy
font = cv2.FONT_HERSHEY_SIMPLEX
prediction_history = deque(maxlen=5)  # Lưu 5 kết quả gần nhất
#############################################

# MỞ CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# TẢI MÔ HÌNH HUẤN LUYỆN
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# HÀM TIỀN XỬ LÝ ẢNH
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Giảm nhiễu
    img = img / 255.0  # Chuẩn hóa
    return img

# HÀM LẤY TÊN BIỂN BÁO
def getClassName(classNo):
    classNames = [
        'toc do toi da 20 km/h', 'toc do toi da 30 km/h', 'toc do toi da 50 km/h',
        'toc do toi da 60 km/h', 'toc do toi da 70 km/h', 'toc do toi da 80 km/h',
        'het gioi han toc do', 'toc do toi da 100 km/h', 'toc do toi da 120 km/h',
        'toc do toi da 40km/h', 'cam oto tai vuot',
        'nhuong duong tai nga tu tiep theo', 'duong uu tien', 'giao nhau voi duong uu tien', 'dung lai',
        'duong cam', 'cam xe tai tren 3.5 tan ', 'cam di nguoc chieu',
        'cam re trai', 'duong cong nguy hiem ben trai', 'duong cong nguy hiem ben phai',
        'duong cong kep', 'duong go ghe', 'duong tron', 'duong hep ben phai',
        'dang thi cong', 'tin hieu giao thong', 'nguoi di bo', 'tre em qua duong',
        'xe dap qua duong', 'canh bao bang tuyet', 'dong vat hoang da qua duong',
        'het moi lenh cam', 're phai phia truoc', 're trai phia truoc',
        'chi duoc di thang', 'di thang hoac re phai', 'di thang hoac re trai', 'di ben phai',
        'di ben trai', 'vong xuyen bat buoc', 'het cam vuot',
        'het cam vuot voi xe tren 3.5 tan'
    ]
    if 0 <= classNo < len(classNames):
        return classNames[classNo]
    else:
        return "Unknown"

# VÒNG LẶP CHÍNH
while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Không đọc được camera")
        break

    # Resize và xử lý ảnh
    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img)

    # Bỏ qua ảnh mờ hoặc tối
    if np.std(img) < 0.05:
        continue

    # Hiển thị ảnh sau xử lý
    img_display = (img * 255).astype(np.uint8)
    cv2.imshow("Processed Image", img_display)

    # Chuẩn bị cho mô hình
    img_input = img.reshape(1, 32, 32, 1)

    # Dự đoán
    predictions = model.predict(img_input)
    classIndex = int(np.argmax(predictions))
    probabilityValue = float(np.amax(predictions))

    if probabilityValue > threshold:
        prediction_history.append(classIndex)
        most_common = Counter(prediction_history).most_common(1)[0][0]
        label = getClassName(most_common)

        # Hiển thị kết quả
        cv2.putText(imgOriginal, f"{most_common} {label}",
                    (50, 40), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"Confidence: {round(probabilityValue*100, 2)}%",
                    (50, 80), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Hiển thị kết quả cuối cùng
    cv2.imshow("Output", imgOriginal)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
