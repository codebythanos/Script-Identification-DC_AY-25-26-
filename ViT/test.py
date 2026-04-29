import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_hub
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_CLASSES  = 12
SEED         = 42
WEIGHTS_PATH = 'best_vit_model.keras'
TEST_DIR     = '/content/dataset/12-way script classification dataset/test_478'

tf.random.set_seed(SEED)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print("TensorFlow:", tf.__version__)

# ── Load test data ────────────────────────────────────────────────────────────
def load_test_dataset(data_dir):
    data_dir   = Path(data_dir)
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    cls_names  = [d.name for d in class_dirs]
    cls_to_idx = {name: idx for idx, name in enumerate(cls_names)}

    all_paths, all_labels = [], []
    for class_dir in class_dirs:
        label = cls_to_idx[class_dir.name]
        for p in class_dir.glob('*'):
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                all_paths.append(str(p))
                all_labels.append(label)

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)

    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    labels_oh = tf.keras.utils.to_categorical(all_labels, NUM_CLASSES)
    ds = (tf.data.Dataset.from_tensor_slices((all_paths, labels_oh))
          .map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))

    return ds, cls_names, all_labels

print("Loading test set...")
test_ds, class_names, true_labels = load_test_dataset(TEST_DIR)
print(f"Test samples : {len(true_labels)}")
print(f"Classes      : {class_names}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model from: {WEIGHTS_PATH}")
model = keras.models.load_model(WEIGHTS_PATH)
print("Model loaded.")

# ── Evaluate ──────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest Loss : {test_loss:.4f}")
print(f"Test Acc  : {test_acc:.4f}  ({int(test_acc*len(true_labels))}/{len(true_labels)})")

# ── Classification report ─────────────────────────────────────────────────────
y_pred = np.argmax(model.predict(test_ds, verbose=1), axis=1)
print("\n" + classification_report(true_labels, y_pred, target_names=class_names, digits=4))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(true_labels, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix — Best ViT | Acc: {test_acc:.4f}', fontsize=14)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('final_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: final_confusion_matrix.png")
