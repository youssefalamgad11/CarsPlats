# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:13:51 2025

@author: youss
"""

import os
import numpy as np
import pytesseract
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw, ImageFont

# تحميل ومعالجة البيانات
def load_data(images_path, labels_path):
    images = []
    labels = []
    for img_file in sorted(os.listdir(images_path)):
        img_path = os.path.join(images_path, img_file)
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(labels_path, label_file)
        
        # التحقق من وجود ملفات الصورة والنص
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} not found, skipping...")
            continue
        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} not found, skipping...")
            continue
        
        # تحميل ومعالجة الصورة
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:  # التحقق من تحميل الصورة بنجاح
            img = cv2.resize(img, (128, 64))  # تغيير حجم الصور لتوحيد المدخلات
            images.append(img)
        else:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        # قراءة ملف النص المرتبط بالصورة
        with open(label_path, 'r', encoding='utf-8') as f:
            label = f.read().strip()
            labels.append(label)
    
    return np.array(images), np.array(labels)

# تهيئة البيانات
images_path = 'plats/Vehicles'
labels_path = 'plats/Vehicles Labeling'
images, labels = load_data(images_path, labels_path)

# تحويل الصور إلى الشكل المطلوب
images = images.reshape(-1, 128, 64, 1) / 255.0

# تحويل التسميات إلى أرقام
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_encoder.classes_))

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# بناء النموذج
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# حفظ النموذج
model.save('vehicle_plate_model.h5')

# تحميل النموذج المدرب
model = tf.keras.models.load_model('vehicle_plate_model.h5')

# دالة لتحميل الصورة من الجهاز
def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # تحويل الصورة إلى تدرجات الرمادي
    img_resized = img.resize((128, 64))
    
    # تحويل الصورة إلى مصفوفة
    img_array = np.array(img_resized).reshape(1, 128, 64, 1) / 255.0
    
    # توقع النموذج
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction, axis=1)
    plate_text = label_encoder.inverse_transform(predicted_label)
    
    # إضافة النص إلى الصورة
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 20)
    draw.text((10, 10), plate_text[0], font=font, fill="red")
    
    # حفظ وعرض الصورة مع النص
    img.save('output.jpg')
    img.show()
    print(f"Detected Plate: {plate_text[0]}")

# تشغيل دالة تحميل الصورة
image_path = input("Enter the path to your image: ")
load_image(image_path)