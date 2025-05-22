# German Traffic Sign Classification with VGG16

A TensorFlow pipeline for classifying German traffic signs (GTSRB) using a pre‑trained VGG16 backbone and custom data preprocessing.

---

## 🚀 Features

* **Dataset**: GTSRB with bounding-box ROI cropping
* **Model**: Pre-trained VGG16 (ImageNet) backbone + custom head
* **Two-Stage Training**:

  1. **Stage A**: Train head layers (freeze VGG16 base)
  2. **Stage B**: Fine-tune last conv block (unfreeze `block5_*`)
* **Augmentation**: Random rotations, shifts, zoom
* **Performance**: Val accuracy \~93.4% after fine-tuning

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
