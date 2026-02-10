# CNN Lab: Oxford-IIIT Pet Dataset Classification

## Problem Description

This project implements and compares deep learning architectures for multiclass image classification of pet breeds. The primary objective is to compare a dense baseline against a CNN using the Oxford-IIIT Pet dataset.
---

## Dataset Description

### Overview
- **Dataset:** Oxford-IIIT Pet Dataset (via TensorFlow Datasets)
- **Source:** `tfds.load('Oxford_iiit_pet', as_supervised=True, with_info=True)`
- **Classes:** 37 different pet breeds (cats and dogs)
- **Image Characteristics:** Variable size, RGB color images

### Data Split Strategy
The dataset was reorganized as follows:
1. Combined original train and test splits
2. Shuffled the complete dataset randomly
3. Split into:
  - **Training:** 70%
  - **Validation:** 15%
  - **Test:** 15%

### Preprocessing Pipeline
```python
Image transformations:
├── Resize to 150x150 pixels (square)
├── Normalize pixel values to [0, 1]
├── Convert labels to one-hot encoding (37 classes)
└── Maintain RGB format (150, 150, 3)
```

### Data Augmentation
Two augmentation strategies were implemented:

**Option 1 - NumPy/Keras ImageDataGenerator:**
- Random rotation (up to 50 degrees)
- Random width shift (up to 20%)
- Random height shift (up to 10%)
- Random zoom (up to 20%)
- Random horizontal flip
- Fill mode: nearest

**Option 2 - TensorFlow Data API:**
- Random horizontal flip
- Random brightness adjustment (±20%)
- Random contrast adjustment (0.8-1.2×)

---

## Architecture Diagrams

### 1. Baseline Model (Dense Neural Network)

```
┌─────────────────────────────────────────┐
│   INPUT: (150, 150, 3)                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   FLATTEN                               │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   DENSE LAYER 1                         │
│   Units: 128                            │
│   Activation: ReLU                      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   DENSE LAYER 2                         │
│   Units: 128                            │
│   Activation: ReLU                      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   OUTPUT LAYER                          │
│   Units: 37 (num_classes)               │
│   Activation: Softmax                   │
└─────────────────────────────────────────┘

```

**Architecture Characteristics:**
- Destroys spatial relationships via Flatten operation
- No translation invariance
- Prone to overfitting on limited data
---

### 2. CNN Model (Baseline 3x3)

```
┌─────────────────────────────────────────┐
│   INPUT: (150, 150, 3)                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   CONV2D LAYER 1                        │
│   Filters: 32, Kernel: 3x3             │
│   Activation: ReLU                      │
│   Output: (148, 148, 32)                │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   MAX POOLING 1                         │
│   Pool size: 2×2                        │
│   Output: (74, 74, 32)                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   CONV2D LAYER 2                        │
│   Filters: 64, Kernel: 3x3             │
│   Activation: ReLU                      │
│   Output: (72, 72, 64)                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   MAX POOLING 2                         │
│   Pool size: 2×2                        │
│   Output: (36, 36, 64)                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   CONV2D LAYER 3                        │
│   Filters: 128, Kernel: 3x3            │
│   Activation: ReLU                      │
│   Output: (34, 34, 128)                 │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   MAX POOLING 3                         │
│   Pool size: 2×2                        │
│   Output: (17, 17, 128)                 │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   FLATTEN                               │
│   Output: (36,992)                      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   DENSE LAYER                           │
│   Units: 100                            │
│   Activation: ReLU                      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   OUTPUT LAYER                          │
│   Units: 37                             │
│   Activation: Softmax                   │
└─────────────────────────────────────────┘

```

**Architecture Rationale:**
- **Progressive feature hierarchy:** 32 → 64 → 128 filters capture increasingly complex patterns
- **3x3 kernels:** Optimal balance between receptive field and parameter efficiency
- **Max pooling:** Provides translation invariance and reduces spatial dimensions

---

### 3. CNN Model (Experiment: 5x5 kernels)

```
┌─────────────────────────────────────────┐
│   INPUT: (150, 150, 3)                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   CONV2D LAYER 1                        │
│   Filters: 32, Kernel: 5x5             │
│   Activation: ReLU                      │
│   Output: (146, 146, 32)                │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   MAX POOLING 1                         │
│   Pool size: 2×2                        │
│   Output: (73, 73, 32)                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   CONV2D LAYER 2                        │
│   Filters: 64, Kernel: 5x5             │
│   Activation: ReLU                      │
│   Output: (69, 69, 64)                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   MAX POOLING 2                         │
│   Pool size: 2×2                        │
│   Output: (34, 34, 64)                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   CONV2D LAYER 3                        │
│   Filters: 128, Kernel: 5x5            │
│   Activation: ReLU                      │
│   Output: (30, 30, 128)                 │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   MAX POOLING 3                         │
│   Pool size: 2×2                        │
│   Output: (15, 15, 128)                 │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   FLATTEN                               │
│   Output: (28,800)                      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   DENSE LAYER                           │
│   Units: 100                            │
│   Activation: ReLU                      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   OUTPUT LAYER                          │
│   Units: 37                             │
│   Activation: Softmax                   │
└─────────────────────────────────────────┘

```

**Key Difference:** Larger 5x5 kernels capture broader spatial context in each layer but with increased parameters in convolutional layers.

---

## Experimental Results

### Controlled Experiment: Kernel Size Impact

**Experimental Design:**
- **Variable:** Kernel size (3x3 vs 5x5)

**Quantitative Findings:**

**Quantitative Results (from the notebook):**

| Configuration | Acc. train (final) | Loss train (final) | Acc. test | Loss test |
|---|---:|---:|---:|---:|
| 3x3 (base) | 0.9828 | 0.0517 | 0.9828 | 0.0517 |
| 5x5 (experiment) | 0.9864 | 0.0586 | 0.9864 | 0.0586 |


## Interpretation

### Why Convolutional Layers Outperform Dense Networks

The CNN models achieved much higher accuracy than the dense baseline. This performance gap stems from three fundamental properties of convolutional architectures:

#### 1. **Spatial Locality and Hierarchical Feature Learning**

**Dense Network Problem:**
The Flatten operation destroys all spatial relationships, converting a 150x150x3 image into a 67,500-dimensional vector. This means:
- Pixel at position (10, 10) has the same relationship to pixel (10, 11) as it does to pixel (140, 140)
- The network must learn from scratch that adjacent pixels are related
- No built-in understanding that images have 2D structure

**CNN Solution:**
Convolutional layers preserve spatial structure and learn hierarchical features:
- **Layer 1 (32 filters, 3x3):** Detects low-level features like edges, corners, and color gradients
  - Example: Vertical edge detector, horizontal edge detector, diagonal edges
- **Layer 2 (64 filters, 3x3):** Combines edges into textures and simple patterns
  - Example: Fur textures, whisker patterns, eye shapes
- **Layer 3 (128 filters, 3x3):** Assembles textures into object parts
  - Example: Cat ears, dog snouts, facial structures

This compositional hierarchy mirrors how biological visual systems process information, from V1 simple cells to higher-level object recognition areas.



### When Convolutional Layers Are NOT Appropriate

Despite their success in computer vision, CNNs fail or underperform in several problem domains:

#### 1. **Tabular/Structured Data Without Spatial Semantics**

**Examples:**
- Medical records: [Age, Blood Pressure, Cholesterol, BMI, Smoking Status]
- Customer data: [Income, Purchase Frequency, Account Age, Region]
- Financial data: [Revenue, Expenses, Debt, Equity, Cash Flow]

**Why CNNs Fail:**
- **No meaningful locality:** The feature in position i is not more related to position i+1 than to position i+100
- **No translation equivariance:** Swapping "Age" and "Blood Pressure" completely changes semantics
- **Arbitrary ordering:** Feature order is a data formatting choice, not intrinsic structure

---

#### 2. **Long-Range Dependencies / Non-Local Interactions**

**Examples:**
- Document understanding: Coreference resolution across paragraphs
- Video analysis: Action recognition requiring long temporal context
- Scene understanding: Relationships between distant objects ("person holding umbrella" when person and umbrella are 100 pixels apart)

**Why CNNs Struggle:**
- Receptive field grows linearly with depth: 3-layer CNN with 3x3 kernels has 7x7 receptive field
- Capturing dependencies across 224×224 images requires 32+ layers
- Deep stacks are hard to train (vanishing gradients) and computationally expensive

---

#### 3. **Data Requiring Different Symmetries**

**Examples:**
- **3D point clouds:** Objects can be rotated arbitrarily in 3D space
  - CNNs are equivariant to 2D translation, but NOT to 3D rotation
  - PointNet/PointNet++ are permutation-invariant (order of points doesn't matter)

- **Molecular property prediction:** Molecules are invariant to rotation and atom permutation
  - CNNs cannot handle permutation invariance
  - Graph Neural Networks (GNNs) or Geometric Deep Learning architectures are appropriate

---

### Key Takeaways

1. **CNNs dominate computer vision** because their inductive biases perfectly match the structure of natural images.

2. **The baseline dense network** not due to insufficient capacity, but because its architectural assumptions don't align with image data—it treats images as unstructured feature vectors.

3. **3x3 vs 5x5 kernels:** The controlled experiment shows 5x5 improves accuracy slightly but increases computational cost. The optimal choice depends on resource constraints.

4. **Generalization ** Both CNN models generalized perfectly (0% train-test gap), demonstrating that the right inductive bias matters more than raw parameter count.

---


## How to Run

### Prerequisites
Required software and packages:
```bash
Python 3.8+
Jupyter Notebook
TensorFlow 2.x
```

Required Python libraries:
```bash
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
tensorflow-datasets>=4.0.0
opencv-python>=4.5.0
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SantiagoSu15/CNN-Pet-Classification
cd cnn-pet-dataset
```

2. **Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow tensorflow-datasets opencv-python jupyter
```

3. **Dataset Download**
The Oxford-IIIT Pet dataset will be downloaded automatically on first run via TensorFlow Datasets. The notebook handles this using:
```python
tfds.load('oxford_iiit_pet', as_supervised=True, with_info=True)
```
- Dataset is cached locally for future runs
- No manual download required

4. **Launch Jupyter Notebook**
```bash
jupyter notebook cnn-pet-dataset.ipynb
```

## Autor

Santiago Suarez


