import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import pickle

# Load model
with open('model_trained.p', 'rb') as file:
    model = pickle.load(file)

# Dictionary to label all traffic signs classes
classes = {
    1: 'tốc độ tối đa (20km/h)', 2: 'tốc độ tối đa (30km/h)', 3: 'tốc độ tối đa (50km/h)',
    4: 'tốc độ tối đa (60km/h)', 5: 'tốc độ tối đa (70km/h)', 6: 'tốc độ tối đa (80km/h)',
    7: 'Hết giới hạn tốc độ',
    8: 'tốc độ tối đa (100km/h)', 9: 'tốc độ tối đa (120km/h)',
    10: 'tốc độ tối đa 40km/h',
    11: 'cấm oto tải vượt', 12: 'Nhường đường tại ngã tư tiếp theo',
    13: 'Đường ưu tiên', 14: 'giao nhau với đường ưu tiên', 15: 'Dừng lại', 16: 'đường cấm',
    17: 'Cấm xe trên 3.5 tấn', 18: 'cấm đi ngược chiều', 19: 'cấm rẽ trái',
    20: 'Đường cong nguy hiểm bên trái', 21: 'Đường cong nguy hiểm bên phải', 22: 'Đường cong kép',
    23: 'Đường gồ ghề', 24: 'Đường trơn', 25: 'Đường hẹp bên phải',
    26: 'Đang thi công', 27: 'Tín hiệu giao thông', 28: 'Người đi bộ',
    29: 'Trẻ em qua đường', 30: 'Xe đạp qua đường', 31: 'Cảnh báo băng tuyết',
    32: 'Động vật hoang dã qua đường', 33: 'Hết mọi lệnh cấm', 34: 'Rẽ phải phía trước',
    35: 'Rẽ trái phía trước', 36: 'Chỉ được đi thẳng', 37: 'Đi thẳng hoặc rẽ phải',
    38: 'Đi thẳng hoặc rẽ trái', 39: 'Đi bên phải', 40: 'Đi bên trái',
    41: 'Vòng xuyến bắt buộc', 42: 'Hết cấm vượt', 43: 'Hết cấm vượt với xe trên 3.5 tấn'
}

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Nhận dạng biển báo giao thông')
top.configure(background='#ffffff')

label = Label(top, background='#ffffff', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Function to classify image
def classify(file_path):
    global label
    image = Image.open(file_path).convert('L')     # Grayscale
    image = image.resize((32, 32))                 # Resize to model input
    image = np.array(image)
    image = image / 255.0                          # Normalize
    image = image[..., np.newaxis]                 # Add channel dimension (32,32,1)
    image = np.expand_dims(image, axis=0)          # Add batch dimension (1,32,32,1)

    pred_probabilities = model.predict(image)[0]
    pred = np.argmax(pred_probabilities)
    sign = classes.get(pred + 1, "Unknown Sign")

    label.configure(foreground='#011638', text=sign)

# Function to show classify button
def show_classify_button(file_path):
    classify_b = Button(top, text='Nhận dạng', command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

# Function to upload image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text='Tải ảnh lên', command=upload_image, padx=10, pady=5)
upload.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text='Nhận dạng biển báo giao thông', pady=10, font=('arial', 20, 'bold'))
heading.configure(background='#ffffff', foreground='#364156')
heading.pack()

top.mainloop()
