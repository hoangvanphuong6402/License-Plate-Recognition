import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from matplotlib import pyplot as plt

from lib_detection import load_model as load_model_detection, detect_lp, im2single

# Load model LP detection
wpod_net_path = "License-Plate-Recognition\wpod-net_update1.json"
wpod_net = load_model_detection(wpod_net_path)

# Load model CNN
model_cnn = load_model('License-Plate-Recognition\cnn_model.h5')


# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts


def maximizeContrast(imgGrayscale):
    # Làm cho độ tương phản lớn nhất
    height, width, _ = imgGrayscale.shape
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # tạo bộ lọc kernel

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement,
                                 iterations=10)  # nổi bật chi tiết sáng trong nền tối
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement,
                                   iterations=10)  # Nổi bật chi tiết tối trong nền sáng

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    # Kết quả cuối là ảnh đã tăng độ tương phản
    return imgGrayscalePlusTopHatMinusBlackHat


# Dinh nghia cac ky tu tren bien so
char_list = '0123456789ABCDEFGHKLMNPRSTUVXYZ'


# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString


# Tạo hộp thoại để chọn tệp ảnh
root = tk.Tk()
root.withdraw()
img_path = filedialog.askopenfilename(initialdir="test/", title="Chọn ảnh", filetypes=(
("Image files", "*.png;*.jpeg;*.jpg;*.gif;*.bmp"), ("all files", "*.*")))

# Kiểm tra xem người dùng đã chọn một tệp ảnh hay chưa
if not img_path:
    print("Không có tệp ảnh được chọn. Kết thúc chương trình.")
    exit()

# Đọc file ảnh đầu vào
Ivehicle = cv2.imread(img_path)

if Ivehicle is None:
    print("Không thể tải ảnh. Hãy kiểm tra lại đường dẫn của ảnh.")
    exit()

# Tiền xử lý ảnh: Tăng độ tương phản
# Ivehicle = maximizeContrast(Ivehicle)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

_, LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

# Cau hinh tham so cho model CNN
digit_w = 30  # Kich thuoc ki tu
digit_h = 60  # Kich thuoc ki tu

if len(LpImg):

    # Chuyen doi anh bien so
    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=255.0)

    cv2.imshow("Detected License Plate", LpImg[0])
    cv2.waitKey(0)

    # LpImg[0] là ảnh biển số đã cắt ra
    roi = LpImg[0]

    # Lấy chiều dài và chiều rộng của ảnh
    height, width = roi.shape[:2]

    # Tính tỷ lệ giữa chiều dài và chiều rộng
    ratio = width / height

    plate_info = ""

    binary_array = []

    if 3.5 < ratio < 6.5:
        print("Biển số có 1 dòng.")
        # Chuyen anh bien so ve gray
        gray = cv2.cvtColor(LpImg[0], cv2.COLOR_BGR2GRAY)

        cv2.imshow("Gray detected License Plate", gray)
        cv2.waitKey(0)

        # Ve histogram cua anh gray truoc khi threshold
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(gray.ravel(), bins=256, color='gray')
        plt.title("Histogram of image before threshold")

        # Ap dung threshold de phan tach so va nen
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # Ve histogram cua anh binary sau khi threshold
        plt.subplot(1, 2, 2)
        plt.hist(binary.ravel(), bins=256, color='black')
        plt.title("Histogram of image after threshold")

        plt.show()

        cv2.imshow("License plate after threshold", binary)
        cv2.waitKey()

        # Segment kí tự
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

        cv2.imshow("After dilation", thre_mor)
        cv2.waitKey(0)

        cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1.5 <= ratio <= 3.5:  # Chon cac contour dam bao ve ratio w/h
                if h / roi.shape[0] >= 0.6:  # Chon cac contour cao tu 60% bien so tro len

                    # Ve khung chu nhat quanh so
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Tach so va predict
                    curr_num = thre_mor[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num, dtype=np.float32)

                    # Dua vao model CNN
                    curr_num = curr_num.reshape(-1, digit_h, digit_w, 1)
                    result = np.argmax(model_cnn.predict(curr_num), axis=-1)
                    result = int(result[0])

                    if result <= 9:  # Neu la so thi hien thi luon
                        result = str(result)
                    else:  # Neu la chu thi chuyen bang ASCII
                        result = chr(result)

                    plate_info += result

        cv2.imshow("Contour found", roi)
        cv2.waitKey()

        # Viet bien so len anh
        cv2.putText(Ivehicle, fine_tune(plate_info), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
    else:
        print("Biển số có 2 dòng.")

        # Chia đôi ảnh biển số thành 2 dòng
        upper_half = roi[:height // 2, :]
        lower_half = roi[height // 2:, :]

        # Chuyển ảnh thành gray
        gray_upper = cv2.cvtColor(upper_half, cv2.COLOR_BGR2GRAY)
        gray_lower = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)

        # Áp dụng threshold để phân tách số và nền
        binary_upper = cv2.threshold(gray_upper, 127, 255, cv2.THRESH_BINARY_INV)[1]
        binary_lower = cv2.threshold(gray_lower, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # Segment kí tự cho từng dòng
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Dòng trên
        thre_mor_upper = cv2.morphologyEx(binary_upper, cv2.MORPH_DILATE, kernel3)
        cont_upper, _ = cv2.findContours(thre_mor_upper, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        plate_info_upper = ""
        for c in sort_contours(cont_upper):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1.5 <= ratio <= 3.5:  # Chọn các contour đảm bảo về tỷ lệ w/h
                if h / upper_half.shape[0] >= 0.6:  # Chọn các contour cao từ 60% của biển số trở lên
                    # Vẽ khung chữ nhật quanh số
                    cv2.rectangle(upper_half, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Tách số và dự đoán
                    curr_num = thre_mor_upper[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num, dtype=np.float32)

                    # Đưa vào model CNN
                    curr_num = curr_num.reshape(-1, digit_h, digit_w, 1)
                    result = np.argmax(model_cnn.predict(curr_num), axis=-1)
                    result = int(result[0])

                    if result <= 9:  # Nếu là số thì hiển thị luôn
                        result = str(result)
                    else:  # Nếu là chữ thì chuyển bằng ASCII
                        result = chr(result)

                    plate_info_upper += result

        # Dòng dưới
        thre_mor_lower = cv2.morphologyEx(binary_lower, cv2.MORPH_DILATE, kernel3)
        cont_lower, _ = cv2.findContours(thre_mor_lower, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        plate_info_lower = ""
        for c in sort_contours(cont_lower):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1.5 <= ratio <= 3.5:  # Chọn các contour đảm bảo về tỷ lệ w/h
                if h / lower_half.shape[0] >= 0.6:  # Chọn các contour cao từ 60% của biển số trở lên
                    # Vẽ khung chữ nhật quanh số
                    cv2.rectangle(lower_half, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Tách số và dự đoán
                    curr_num = thre_mor_lower[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num, dtype=np.float32)

                    # Đưa vào model CNN
                    curr_num = curr_num.reshape(-1, digit_h, digit_w, 1)
                    result = np.argmax(model_cnn.predict(curr_num), axis=-1)
                    result = int(result[0])

                    if result <= 9:  # Nếu là số thì hiển thị luôn
                        result = str(result)
                    else:  # Nếu là chữ thì chuyển bằng ASCII
                        result = chr(result)

                    plate_info_lower += result

        plate_info = plate_info_upper + plate_info_lower
        cv2.imshow("Contour found", roi)
        cv2.waitKey()

        # Viết biển số lên ảnh
        cv2.putText(Ivehicle, fine_tune(plate_info), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)

    # Hiển thị ảnh
    print("License Plate = ", plate_info)
    cv2.imshow("Result", Ivehicle)
    cv2.waitKey()

cv2.destroyAllWindows()
