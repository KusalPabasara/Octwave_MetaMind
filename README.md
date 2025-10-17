# Spot the Difference Challenge - Solution

## 📋 Competition Overview

**Task**: Identify differences between two images (continuity errors in filmmaking)

**Challenge Categories**:
1. **Added Objects**: Objects present in image 2 but not in image 1
2. **Removed Objects**: Objects present in image 1 but not in image 2  
3. **Position-Changed Objects**: Objects that moved or changed appearance

**Evaluation Metric**: Mean F1 Score across the three categories (added, removed, changed)

## 🎯 Solution Strategy

### Approach: Siamese Deep Learning Model

Our solution uses a **dual-branch Siamese network** that:

1. **Extracts features** from both images independently using a pre-trained CNN backbone
2. **Compares features** using difference operations (absolute difference, concatenation, element-wise product)
3. **Predicts multi-labels** for each category using separate classification heads

### Model Architecture

```
Input: Image Pair (img_1, img_2)
    ↓
[Shared CNN Backbone - EfficientNet-B3/ResNet18]
    ↓
[Feature Extraction & Processing]
    ↓
[Difference Computation]
    ↓
[Three Classification Heads]
    ↓
Output: (added_objs, removed_objs, changed_objs)
```

### Key Features

- **Pre-trained Backbone**: Uses EfficientNet-B3 (advanced) or ResNet18 (baseline) pre-trained on ImageNet
- **Multi-label Classification**: Each category can have multiple objects
- **Siamese Architecture**: Processes both images with shared weights
- **Multiple Difference Operations**: Captures various types of changes
- **Data Augmentation**: Horizontal flips, brightness/contrast adjustments (carefully applied to maintain consistency)

## 📁 Files Structure

```
spot-the-difference-challenge/
├── data/
│   └── data/
│       ├── 30000_1.png, 30000_2.png
│       ├── 30001_1.png, 30001_2.png
│       └── ... (6018 image pairs)
├── train.csv              # Training labels (4536 samples)
├── test.csv               # Test set (1482 samples)
├── solution.py            # Advanced solution with EfficientNet-B3
├── baseline_solution.py   # Faster baseline with ResNet18
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── submission.csv        # Output predictions (generated)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
- torch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python >= 4.8.0
- pandas, numpy, scikit-learn
- albumentations (for augmentation)
- tqdm (for progress bars)

### 2. Run Training

**Option A: Baseline Solution (Faster, ~30-45 min on GPU)**
```bash
python baseline_solution.py
```
- Uses ResNet18 backbone
- Image size: 224x224
- Epochs: 10
- Lighter and faster for quick experimentation

**Option B: Advanced Solution (Better accuracy, ~1-2 hours on GPU)**
```bash
python solution.py
```
- Uses EfficientNet-B3 backbone
- Image size: 384x384
- Epochs: 20
- More parameters, better feature extraction

### 3. Output

Both scripts will:
1. Load and preprocess the data
2. Train the model with validation split (85-15)
3. Save the best model based on validation loss
4. Generate predictions on test set
5. Create `submission.csv` with predictions

## 📊 Data Analysis

### Training Data Statistics
- **Total samples**: 4,536 image pairs
- **Format**: Each row has img_id and three label columns
- **Common objects**: person, car, vehicle, bicycle, man, guy, group, box, chair, etc.
- **Label format**: Space-separated strings (e.g., "person vehicle" or "none")

### Sample Training Data
```csv
img_id,added_objs,removed_objs,changed_objs
35655,none,none,none
30660,none,person vehicle,none
34838,man person,car person,none
34045,person,none,car
```

## 🔧 Hyperparameters

### Baseline Solution
- Image Size: 224x224
- Batch Size: 16
- Learning Rate: 1e-3
- Epochs: 10
- Model: ResNet18

### Advanced Solution
- Image Size: 384x384
- Batch Size: 8
- Learning Rate: 1e-4
- Epochs: 20
- Model: EfficientNet-B3
- Scheduler: CosineAnnealingLR
- Dropout: 0.2-0.3

## 📈 Model Training Details

### Loss Function
- **BCEWithLogitsLoss**: Binary cross-entropy for multi-label classification
- **Total Loss**: Sum of losses from all three heads (added + removed + changed)

### Optimization
- **Optimizer**: AdamW (advanced) / Adam (baseline)
- **Weight Decay**: 1e-4 (for regularization)
- **Learning Rate Scheduler**: Cosine annealing for smooth convergence

### Validation Strategy
- **Split**: 85% training, 15% validation
- **Early Stopping**: Save best model based on validation loss
- **Threshold**: 0.3 for converting probabilities to binary predictions

## 🎲 Data Augmentation

Applied carefully to maintain image pair consistency:
- Resize to target size
- Horizontal flip (p=0.3)
- Random brightness/contrast (p=0.2)
- Normalization (ImageNet stats)

## 💡 Key Implementation Details

### Multi-Label Encoding
Uses `MultiLabelBinarizer` from scikit-learn to:
- Convert space-separated strings to binary vectors
- Handle multiple objects per category
- Support "none" as a valid class

### Prediction Threshold
- Default: 0.3 (can be tuned for better precision/recall trade-off)
- Lower threshold → more predictions (higher recall)
- Higher threshold → fewer predictions (higher precision)

### Difference Features
Three types of difference computations:
1. **Concatenation**: [feat1, feat2] - captures both images
2. **Absolute Difference**: |feat1 - feat2| - captures changes
3. **Element-wise Product**: feat1 * feat2 - captures similarities

## 🔍 Evaluation Metric

**Mean F1 Score** calculation:

For each category (added, removed, changed):
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Final Score:
```
Mean F1 = (F1_added + F1_removed + F1_changed) / 3
```

## 📝 Submission Format

The output `submission.csv` has the same format as test.csv:
```csv
img_id,added_objs,removed_objs,changed_objs
34478,none,none,none
32209,person,none,car vehicle
34741,none,car,person
...
```

## 🎯 Expected Performance

### Baseline Solution (ResNet18)
- **Training Time**: ~30-45 minutes on GPU
- **Expected CV Score**: ~0.45-0.55 F1
- **Advantages**: Fast iteration, good for experimentation

### Advanced Solution (EfficientNet-B3)
- **Training Time**: ~1-2 hours on GPU
- **Expected CV Score**: ~0.55-0.65 F1
- **Advantages**: Better feature extraction, higher capacity

## 🚀 Improvement Ideas

1. **Ensemble Models**: Combine predictions from multiple models
2. **Test-Time Augmentation (TTA)**: Average predictions from augmented versions
3. **Larger Backbones**: EfficientNet-B4/B5 for better features
4. **Object Detection**: Use YOLO/Faster R-CNN to detect objects first
5. **Attention Mechanisms**: Add spatial attention to focus on differences
6. **Threshold Tuning**: Optimize per-category thresholds on validation set
7. **Pseudo-Labeling**: Use confident test predictions for semi-supervised learning
8. **Cross-Validation**: 5-fold CV for more robust estimates

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size (try 4 or 2)
- Reduce image size (try 224 or 192)
- Use gradient accumulation

### Low Accuracy
- Train for more epochs
- Try different learning rates (1e-5 to 1e-3)
- Adjust prediction threshold
- Add more augmentation

### Slow Training
- Use baseline solution
- Reduce image size
- Increase batch size if memory allows
- Use mixed precision training (add `torch.cuda.amp`)

## 📚 References

- **Competition**: Spot the Difference Challenge (Kaggle)
- **EfficientNet Paper**: Tan & Le, 2019
- **ResNet Paper**: He et al., 2015
- **Multi-Label Classification**: Read et al., 2011

## 👨‍💻 Usage Tips

1. **Start with baseline**: Quick feedback loop
2. **Monitor overfitting**: Watch train vs validation loss
3. **Tune threshold**: Try values from 0.2 to 0.5
4. **Check predictions**: Visualize sample predictions to understand model behavior
5. **GPU recommended**: CPU training will be very slow (10-20x slower)

## 📞 Support

For issues or questions:
1. Check error messages in console output
2. Verify data paths are correct
3. Ensure all dependencies are installed
4. Check GPU availability with `torch.cuda.is_available()`

---

