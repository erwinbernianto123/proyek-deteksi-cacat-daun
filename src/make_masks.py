import os, cv2, numpy as np, pandas as pd

os.makedirs("data/masks", exist_ok=True)
df = pd.read_csv("data/metadata.csv")
missing = []
for f in df.file:
    stem = os.path.splitext(f)[0]
    p = f"data/paint/{stem}.png"
    if not os.path.exists(p):
        missing.append(f)
        continue
    img = cv2.imread(p)
    b, g, r = (img[..., i].astype(int) for i in range(3))
    m = ((r > 200) & (b > 200) & (g < 90)).astype(np.uint8) * 255
    orig = cv2.imread(f"data/images/{f}")
    if m.shape != orig.shape[:2]:
        m = cv2.resize(m, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"data/masks/{stem}.png", m)
    print(f, "piksel cacat:", int((m > 0).sum()))
print("\nBelum ada file PNG untuk:", missing)