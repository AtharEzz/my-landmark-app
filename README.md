# Landmark Classifier - CNN from Scratch vs Transfer Learning

## The Problem

Photo sharing and storage services benefit enormously from location data attached to uploaded images. While many photos carry GPS metadata, a significant portion do not - either because the camera lacked GPS capability or because metadata was scrubbed for privacy reasons.

One way to infer location from a photo is to detect and classify a recognizable landmark in the image. But with hundreds of thousands of landmarks worldwide and millions of photos uploaded daily, human review is completely unscalable.

This project builds an automated solution: a CNN-powered app that predicts the most likely landmark depicted in any user-supplied image, enabling photo services to automatically suggest location tags and organize photos - without requiring GPS metadata.

## Solution

End-to-end CNN pipeline covering data preprocessing, model design, training, comparison, and deployment - completed as part of the **AI Programming with Python Nanodegree (Udacity)**.

**Live App:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AtharEzz/my-landmark-app/580bf04c9dd06a9d6288d8c6fefeb45e8d1417fa?urlpath=voila/render/app/app.ipynb)

---

## Results

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Custom CNN (from scratch) | 51% (644/1250) | 1.90 |
| ResNet50 Transfer Learning | **78% (978/1250)** | **0.87** |

**Transfer learning significantly outperformed the custom CNN** - a result that makes intuitive sense: ResNet50's ImageNet-pretrained features encode rich visual representations (edges, textures, shapes) that transfer well to landmark recognition. The custom CNN, trained from scratch on a relatively small dataset, could not develop the same feature depth in the same number of epochs.

The ResNet50 model was exported via **TorchScript** and deployed as the production model in the live app.

---

## Dataset

- **Source:** Subset of Google Landmarks Dataset v2
- **Classes:** 50 landmark categories from across the world
- **Training set:** ~6,250 images
- **Test set:** 1,250 images
- **Input size:** 224×224 RGB (ResizeCrop for training, CenterCrop for validation/test)

---

## Approach

### Part 1: Custom CNN from Scratch (`cnn_from_scratch.ipynb`)

**Architecture:**
- 4 convolutional blocks, each with 2 Conv layers + BatchNorm + ReLU + MaxPooling
- BatchNormalization after each Conv layer to stabilize training
- Dropout (0.4) before the classifier to reduce overfitting
- Fully connected classifier head for 50 output classes

**Training setup:**
- Optimizer: Adam (lr=0.0001)
- Epochs: 50
- Data augmentation on training set (random horizontal flip, random crop)
- No augmentation on validation/test (center crop only)

**Result:** 51% test accuracy - respectable for a 50-class problem trained from scratch on limited data, but clearly limited by the absence of pretrained features.

### Part 2: Transfer Learning (`transfer_learning.ipynb`)

**Architecture:**
- Base: ResNet50 pretrained on ImageNet (backbone frozen - `requires_grad = False`)
- Head: replaced final classification layer with a new fully connected layer for 50 classes
- Only the new head was trained; backbone weights were not updated

**Training setup:**
- Optimizer: Adam (lr=0.0001, weight_decay=0.0005)
- Epochs: 50
- Same data loading pipeline as Part 1

**Result:** 78% test accuracy - a 27 percentage point improvement over the custom CNN, confirming that frozen pretrained features from ImageNet transfer effectively to landmark classification.

### Part 3: App Deployment (`app.ipynb`)

- Best model (ResNet50) exported via **TorchScript** for portable, framework-independent deployment
- Confusion matrix generated across all 1,250 test predictions to visualize per-class performance
- Deployed as an interactive **Voilà app** accessible via Binder - users can upload any image and receive the top predicted landmarks with confidence scores

---

## Project Structure

```
├── cnn_from_scratch.ipynb     # Custom CNN design, training, and export
├── transfer_learning.ipynb    # ResNet50 transfer learning and export
├── app/
│   └── app.ipynb              # Voilà app for user-facing inference
├── src/
│   ├── data.py                # Data loading and augmentation
│   ├── model.py               # Custom CNN architecture
│   ├── transfer.py            # Transfer learning model setup
│   ├── train.py               # Training, validation, and test loops
│   ├── optimization.py        # Loss and optimizer setup
│   └── predictor.py           # TorchScript export and inference
├── checkpoints/
│   ├── best_val_loss.pt       # Best custom CNN weights
│   └── transfer_exported.pt   # Exported ResNet50 TorchScript model
└── requirements.txt
```

---

## Tools & Libraries

Python, PyTorch, torchvision (ResNet50), TorchScript, Voilà, Binder, NumPy, Matplotlib

---


