import os
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier

# === Konfigurasi Path ===
MODEL_PATH = 'Resources/Model/converted_model.keras'
LABELS_PATH = 'Resources/Model/labels.txt'
ARROW_IMG_PATH = 'Resources/arrow.png'
BACKGROUND_IMG_PATH = 'Resources/background.png'
WASTE_FOLDER = 'Resources/Waste'
BINS_FOLDER = 'Resources/Bins'

# === Inisialisasi Kamera ===
def initialize_camera():
    for i in range(3):  # Coba sampai 3 index
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Kamera ditemukan di index: {i}")
            return cap
        cap.release()
    print("❌ Tidak ada kamera yang tersedia.")
    exit()

# === Load Gambar dalam Folder ===
def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images

# === Pemetaan Label ke Tempat Sampah ===
class_mapping = {
    0: None,   # Background
    1: 0,      # Recyclable
    2: 0,
    3: 3,      # Residual
    4: 3,
    5: 1,      # Hazardous
    6: 1,
    7: 2,      # Food
    8: 2
}

# === MAIN ===
def main():
    cap = initialize_camera()

    classifier = Classifier(MODEL_PATH, LABELS_PATH)
    img_arrow = cv2.imread(ARROW_IMG_PATH, cv2.IMREAD_UNCHANGED)
    img_waste_list = load_images_from_folder(WASTE_FOLDER)
    img_bins_list = load_images_from_folder(BINS_FOLDER)

    class_id_bin = 0

    while True:
        success, img = cap.read()
        if not success:
            print("⚠️ Frame kamera tidak terbaca. Cek koneksi kamera.")
            continue

        try:
            img_resize = cv2.resize(img, (454, 340))
        except Exception as e:
            print(f"❌ Resize error: {e}")
            continue

        img_background = cv2.imread(BACKGROUND_IMG_PATH)

        prediction = classifier.getPrediction(img)
        class_id = prediction[1]
        print(f"Class ID: {class_id}")

        if class_id != 0:
            if class_id - 1 < len(img_waste_list):
                img_background = cvzone.overlayPNG(img_background, img_waste_list[class_id - 1], (909, 127))
            img_background = cvzone.overlayPNG(img_background, img_arrow, (978, 320))
            class_id_bin = class_mapping.get(class_id, 0)

        if class_id_bin < len(img_bins_list):
            img_background = cvzone.overlayPNG(img_background, img_bins_list[class_id_bin], (895, 374))

        # Tempel gambar kamera ke background
        img_background[148:148 + 340, 159:159 + 454] = img_resize

        # Tampilkan ke layar
        cv2.imshow("Waste Classifier Output", img_background)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
