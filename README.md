# Generating Machine Learning Models Using Machine Learning Models

> **Learning new tasks without new data.**  
> This project explores how *machine learning models themselves* can be generated using **other machine learning models**, without direct task-specific supervision.

---

## Overview

This project introduces an unconventional but powerful idea:  
**using generative models to create new classifiers by fusing existing ones**.

Specifically, we leverage **CycleGANs** to merge the learned feature representations of two independently trained **Convolutional Neural Networks (CNNs)**:

- One CNN trained to recognize **cats**
- One CNN trained to recognize the **color black**

By translating and combining their feature spaces, we generate a **new CNN** capable of detecting **black cats** — **without ever being trained on a single black cat image**.

 **Result:**  
The generated model achieves **88% classification accuracy** on black cat detection, despite zero direct exposure to black cat data.

This work opens new directions for:
- Learning under **data scarcity**
- **Automated model generation**
- Knowledge transfer beyond traditional fine-tuning

---

## Key Contributions

-  **CycleGAN-based Model Fusion**  
  Uses CycleGANs to translate and align feature kernels between CNNs trained on unrelated domains.

-  **Generated CNNs (Zero-shot Task Creation)**  
  Constructs a task-specific classifier purely from pre-trained models.

-  **Feature Space Validation**  
  Employs **UMAP**, **K-Means**, and **DBSCAN** to analyze and validate learned representations.

-  **Unsupervised Generalization**  
  Demonstrates black cat recognition without labeled black cat data.

---

##  Core Idea (Intuition)

Instead of training a model on *data*, we train a **model on other models**.

1. Train two CNNs on separate concepts  
   - *Object*: cat  
   - *Attribute*: black  

2. Extract convolutional kernels from both networks.

3. Train **CycleGANs** to translate kernels between these feature domains.

4. Initialize a new CNN using the **CycleGAN-generated kernels**.

5. Evaluate whether this synthesized CNN can recognize *black cats*.

 **It can.**

---

##  Methodology

###  Datasets

| Dataset | Samples |
|------|--------|
| Black / Random Images | 1,826 (1,745 black, 81 random) |
| Cat / Random Images | 30,405 (29,843 cats, 562 random) |
| Kernel Sets for CycleGAN | 4,498 per convolutional layer |

---

###  Model Architectures

#### CNNs
- 2 Convolutional layers
- Kernel size: **5×5**
- Activation: **ReLU**
- Max-pooling layers
- Trained independently on separate domains

#### CycleGANs
- Generator–Discriminator architecture
- Learns **kernel-space translation**, not image translation
- Operates directly on convolutional filters

#### Generated CNN
- Initialized entirely using **CycleGAN-generated kernels**
- No gradient updates using black cat images

---

### 📊 Evaluation Metrics

- **Accuracy**
- **Precision & Recall**
- **Cluster Entropy**
- **Cluster Purity**
- **Cosine Similarity**
- **UMAP Visualization**

---

##  Results

-  The generated CNN successfully **clusters black cat images**
-  UMAP projections show **clear separation** of semantic concepts
-  Cosine similarity confirms meaningful feature alignment
-  Demonstrates **unsupervised semantic composition**

> The model learns *“black AND cat”* without ever seeing a black cat.

---

##  Why This Matters

Traditional ML assumes:
> *New task ⇒ new labeled data*

This project challenges that assumption by showing:
- Tasks can be **composed**
- Models can be **generated**, not trained
- Generative models can operate in **parameter space**, not just data space

This has implications for:
- Low-resource domains
- Privacy-sensitive data
- Automated ML systems
- Foundation model composition

---

##  Future Work

-  Hyperparameter optimization for clustering and feature fusion
-  Alternative feature-space similarity metrics
-  Semantic-aware end-to-end pipelines
-  Scaling to deeper CNNs and transformers
-  Multi-attribute model composition

---

##  Setup & Execution

###  Hardware Requirements
- GPU with **≥ 6GB VRAM** (recommended)

###  Software Requirements

- Python **3.11**
- PyTorch **2.5**
- torchvision
- numpy
- scikit-learn
- seaborn
- matplotlib
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```
### License
MIT

###  Research focus

Generative models, representation learning, and non-traditional ML paradigms.

_“Why train on more data when you can train on more models?”_
