import os, random, cv2, pandas as pd

RAW = r"C:\Users\Acer\Documents\foto-daun-mentah"

random.seed(42)
IMG, PAINT = "data/images", "data/paint"
os.makedirs(IMG, exist_ok=True)
os.makedirs(PAINT, exist_ok=True)
if os.path.exists("data/metadata.csv"):
    raise SystemExit("metadata.csv sudah ada. Hapus dulu jika ingin mengulang.")

rows = []
for kond in sorted(os.listdir(RAW)):
    folder = os.path.join(RAW, kond)
    if not os.path.isdir(folder):
        continue
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.shuffle(files)
    n_dev = round(0.6 * len(files))
    for i, f in enumerate(files, 1):
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            print("GAGAL dibaca:", f)
            continue
        s = min(1.0, 1024 / max(img.shape[:2]))
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        name = f"{kond}_{i:03d}.jpg"
        for d in (IMG, PAINT):
            cv2.imwrite(os.path.join(d, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(dict(file=name, kondisi=kond,
                         sumber="publik" if f.lower().startswith("pub_") else "sendiri",
                         split="dev" if i <= n_dev else "test", asli=f))

df = pd.DataFrame(rows)
df.to_csv("data/metadata.csv", index=False)
print(df.groupby(["kondisi", "split"]).size())
print(df["sumber"].value_counts())