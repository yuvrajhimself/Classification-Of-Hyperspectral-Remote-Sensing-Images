import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from spectral import open_image
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from joblib import dump
from skimage import exposure

print(" Starting HSI Classification...")

# === Load AVIRIS HSI & Ground Truth ===
hsi = open_image(r'C:\Users\HP\Downloads\ahmedabad\AVIRIS_NG_Ahmedabad.hdr').load().astype(np.float32)
gt = open_image(r'C:\Users\HP\Downloads\ahmedabad\AVIRIS_NG_Ahmedabad_Classified.hdr').read_band(0).astype(np.uint8)

assert hsi.shape[:2] == gt.shape, " Shape mismatch between HSI and GT image"

# === Show unique GT classes ===
classes, counts = np.unique(gt, return_counts=True)
print("GT Classes & Counts:")
for c, count in zip(classes, counts):
    print(f"Class {c}: {count} pixels")

# === Mask valid classes: skip Unclassified ===
mask = (gt > 0) & (gt <= 9)
X = hsi[mask]
a?.,mnbcx
y = gt[mask] - 1  

print(f"Using {X.shape[0]} valid samples")

# === FCC with proper contrast stretch ===
def false_color_composite(hsi_cube, bands=[64, 44, 25]):
    """
    FCC: NIR-Red-Green → RGB channels.
    ENVI-like: use 2–98 percentile stretch.
    """
    img = hsi_cube[:, :, bands].astype(np.float32)
    img_stretched = np.zeros_like(img)
    for i in range(3):
        band = img[:, :, i]
        p2, p98 = np.percentile(band, (2, 98))
        img_stretched[:, :, i] = exposure.rescale_intensity(band, in_range=(p2, p98))
    img_stretched = np.clip(img_stretched, 0, 1)
    return img_stretched

fcc_image = false_color_composite(hsi, bands=[64, 44, 25])

# === Also PCA RGB ===
def pca_rgb(hsi_cube):
    reshaped = hsi_cube.reshape(-1, hsi_cube.shape[-1])
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(reshaped)
    rgb = rgb.reshape(hsi_cube.shape[0], hsi_cube.shape[1], 3)
    rgb_min = rgb.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb.max(axis=(0, 1), keepdims=True)
    return (rgb - rgb_min) / (rgb_max - rgb_min)

rgb_image = pca_rgb(hsi)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Downsample for speed
X_train, y_train = resample(X_train, y_train, n_samples=5000, random_state=42)
print(f"📦 Training with {X_train.shape[0]} samples")

# === Train RF ===
clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
print("✅ RF Trained")

# === Predict entire image ===
flat_hsi = hsi.reshape(-1, hsi.shape[-1])
classified_flat = np.zeros(flat_hsi.shape[0], dtype=np.uint8)
classified_flat[mask.flatten()] = clf.predict(X) + 1 
classified_img = classified_flat.reshape(hsi.shape[:2])

# === Save model ===
dump(clf, r'C:\Users\HP\Documents\rf_model.joblib')

# === class names & colors ===
class_names = [
    'lake',        # 1
    'river',       # 2
    'grass',       # 3
    'vegetation',  # 4
    'chinamosaic', # 5
    'tinshed',     # 6
    'concrete',    # 7
    'asphalt',     # 8
    'bareground'   # 9
]

# ENVI RGB data, normalized to 0–1:
colors = [
    (0/255, 139/255, 139/255),  # lake
    (0/255, 0/255, 255/255),    # river
    (0/255, 255/255, 0/255),    # grass
    (0/255, 139/255, 0/255),    # vegetation
    (255/255, 255/255, 0/255),  # chinamosaic
    (0/255, 255/255, 255/255),  # tinshed
    (205/255, 0/255, 0/255),    # concrete
    (0/255, 0/255, 0/255),      # asphalt
    (218/255, 112/255, 214/255) # bareground
]

cmap = ListedColormap(colors)

# === Plot FCC ===
plt.figure(figsize=(12, 6))
plt.imshow(fcc_image)
plt.title("False Color Composite (NIR-Red-Green))")
plt.axis("off")
plt.tight_layout()
plt.show()

# === Plot Classified ===
plt.figure(figsize=(12, 6))
plt.imshow(classified_img, cmap=cmap, vmin=1, vmax=9)
plt.title(" Classified Output ")
plt.axis("off")

legend_patches = [Patch(facecolor=colors[i], label=class_names[i]) for i in range(len(class_names))]
plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
plt.tight_layout()
plt.show()

import rasterio
from rasterio.transform import from_origin
from spectral import envi

# === Read original image  ===
hdr_path = r'C:\Users\HP\Downloads\ahmedabad\AVIRIS_NG_Ahmedabad.hdr'

# Using spectral to read header — but to get transform, better read with GDAL/rasterio
dataset = envi.open(hdr_path)
width, height = dataset.ncols, dataset.nrows

transform = from_origin(72.5, 23.05, 0.0001, 0.0001)  # Example: left, top, xres, yres
crs = 'EPSG:4326'  

# === Prepare GeoTIFF output ===
output_path = r'C:\Users\HP\Documents\classified_output.tif'

with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=classified_img.shape[0],
    width=classified_img.shape[1],
    count=1,
    dtype=classified_img.dtype,
    crs=crs,
    transform=transform
) as dst:
    dst.write(classified_img, 1)

print(f"✅ GeoTIFF saved: {output_path}")

print("press X")
