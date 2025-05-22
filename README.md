# German Traffic Sign Classification with VGG16

A TensorFlow pipeline for classifying German traffic signs (GTSRB) using a pre‑trained VGG16 backbone and custom data preprocessing.

---

## 🚀 Features

* **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark) from Kaggle
* **Preprocessing**: ROI cropping, resizing to 224×224, pixel normalization
* **Augmentation**: Rotation, zoom, width/height shifts during training
* **Model**: Transfer learning with VGG16 (ImageNet weights), two-stage training:

  1. Freeze convolutional base, train head only
  2. Unfreeze last block, fine‑tune full model
* **Callbacks**: Early stopping on validation loss (patience=3)
* **Outputs**: Trained model saved as `gtsrb_finetuned_model.keras`

---

## 📦 Requirements

* Python 3.7+
* `tensorflow` (>=2.5)
* `opencv-python`
* `numpy`
* `pandas`
* `kagglehub` (for dataset download)

Install via:

```bash
pip install tensorflow opencv-python numpy pandas kagglehub
```

---

## 📝 Usage

1. **Download dataset**

   ```python
   import kagglehub
   path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
   ```

2. **Run training**

   ```bash
   python main.py
   ```

   * Stage A: trains classification head (10 epochs)
   * Stage B: fine‑tunes last VGG16 block (10 epochs)
   * Early stopping restores best weights

3. **Evaluate & save**

   * Validation accuracy printed at end
   * Model saved as `gtsrb_finetuned_model.keras`

---

## 📂 File Structure

```plaintext
├── Main.py                     # Preprocessing, model definition, training
├── Train.csv                   # GTSRB annotations (downloaded)
├── gtsrb_finetuned_model.keras # Saved fine‑tuned model
└── logs/                       # (optional) TensorBoard logs
```

---

## 🔧 Customization

* **IMAGE\_SIZE**: Change `IMAGE_SIZE` in `main.py` for different input resolution
* **BATCH\_SIZE/EPOCHS**: Adjust constants at top of `main.py`
* **Augmentation**: Modify `ImageDataGenerator` parameters for stronger transforms
* **Model**: Swap VGG16 for other backbones (e.g., ResNet50) by updating import and `build_model()`

---

## 📄 License

This project is licensed under MIT. See [LICENSE](LICENSE) for details.

---

*Happy road‑sign detecting!*
