import cv2, os, shutil

RAW = r"C:\Users\Acer\Documents\foto-daun-semua"
OUT = r"C:\Users\Acer\Documents\foto-daun-mentah"
T_GELAP, T_TERANG = 60, 110

for k in "ABC":
    os.makedirs(os.path.join(OUT, k), exist_ok=True)

n = {"A": 0, "B": 0, "C": 0}
for f in sorted(os.listdir(RAW)):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = cv2.imread(os.path.join(RAW, f), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    v = img.mean()
    k = "C" if v < T_GELAP else ("B" if v < T_TERANG else "A")
    shutil.copy(os.path.join(RAW, f), os.path.join(OUT, k, f))
    n[k] += 1
print("Jumlah per kondisi:", n)