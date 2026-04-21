import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from spectral import open_image
import rasterio
from skimage.transform import resize

# === Load Data ===
def load_data():
    X = open_image(r'C:\Users\HP\Documents\SOLIDWORKS Downloads\hsi_processed_subset.hdr').load().astype(np.float32)
    with rasterio.open(r'C:\Users\HP\Downloads\TrainingGT\2018_IEEE_GRSS_DFC_GT_TR.tif') as src:
        y = src.read(1)
    y_resized = resize(y, (X.shape[0], X.shape[1]), order=0, preserve_range=True, anti_aliasing=False).astype(int)
    return X, y_resized

# === PCA ===
def applyPCA(X, n=30):
    flat = X.reshape(-1, X.shape[2])
    pca = PCA(n_components=n, whiten=True)
    new = pca.fit_transform(flat)
    return new.reshape(X.shape[0], X.shape[1], n)

# === Patches ===
def create_patches(X, y):
    Xf = X.reshape(-1, X.shape[2])
    yf = y.flatten()
    mask = yf > 0
    return Xf[mask], yf[mask] - 1

# === Rare Class Filter ===
def filter_rare(X, y):
    c = Counter(y)
    keep = [k for k in c if c[k] >= 2]
    mask = np.isin(y, keep)
    return X[mask], y[mask]

# === Oversample ===
def oversample(X, y):
    newX, newY = [], []
    for cls in np.unique(y):
        x_cls = X[y == cls]
        reps = int(np.max(np.bincount(y)) / len(x_cls))
        newX.append(np.tile(x_cls, (reps, 1)))
        newY.append(np.tile(cls, reps))
    finalX = np.concatenate(newX)
    finalY = np.concatenate(newY)
    idx = np.random.permutation(len(finalY))
    return finalX[idx], finalY[idx]

# === Load & Process ===
X, y = load_data()
print("HSI:", X.shape, "| Labels:", y.shape)

rgb = np.stack([X[:, :, i] for i in [10, 20, 30]], axis=-1)
plt.imshow(rgb / np.max(rgb))
plt.title("False RGB")
plt.axis("off")
plt.show()

X_pca = applyPCA(X, n=30)
X_flat, y_flat = create_patches(X_pca, y)
X_flat, y_flat = filter_rare(X_flat, y_flat)
n_classes = len(np.unique(y_flat))

X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.25, stratify=y_flat, random_state=42)
X_train, y_train = oversample(X_train, y_train)

X_train = X_train.reshape(-1, X_train.shape[1], 1)
y_train = to_categorical(y_train, num_classes=n_classes)

# === CNN ===
model = Sequential([
    Conv1D(20, 3, activation='relu', input_shape=(30, 1), padding='same'),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer=SGD(1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

# === Predict Full Image ===
height, width, bands = X_pca.shape
output = np.zeros((height, width), dtype=np.uint8)

for i in range(height):
    row = X_pca[i].reshape(width, bands, 1)
    preds = model.predict(row, verbose=0)
    output[i] = np.argmax(preds, axis=1)

# === Save & Show Output ===
plt.figure(figsize=(10, 5))
plt.imshow(output, cmap='nipy_spectral')
plt.title("Classified Output (with labels)")
plt.axis("off")
plt.colorbar(label='Class ID')
plt.savefig("classified_output_with_labels.png", dpi=300)
plt.show()
