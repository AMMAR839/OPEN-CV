import cv2
import numpy as np

# Fungsi untuk mendeteksi lingkaran
def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100)
    return circles

# Fungsi untuk mendeteksi kotak persegi
def detect_squares(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Jika terdapat 4 sudut
            squares.append(approx)

    return squares

# Fungsi utama
def main():
    # Membaca gambar
    image = cv2.imread('2.jpeg')  # Ganti 'image.png' dengan path ke gambar Anda
    if image is None:
        print("Gambar tidak ditemukan!")
        return

    # Deteksi kotak persegi
    squares = detect_squares(image.copy())  # Menggunakan salinan gambar asli untuk deteksi kotak
    for square in squares:
        cv2.polylines(image, [square], isClosed=True, color=(0, 0, 255), thickness=2)

    # Deteksi lingkaran
    circles = detect_circles(image)
    if circles is not None:
        for circle in circles[0]:
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
            # Menggambar lingkaran
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            # Menggambar titik pusat
            cv2.drawMarker(image, (x, y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            # Menambahkan bounding box
            cv2.rectangle(image, (x - r, y - r), (x + r, y + r), (255, 0, 0), 2)

    # Menampilkan gambar
    cv2.imshow('Hasil Deteksi', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
