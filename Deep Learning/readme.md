# Deep Learning & Core ML Architecture Refresher (ML2)

This repository serves as a comprehensive production-grade refresher tracking my journey from foundational machine learning mathematics to advanced structural paradigms in Computer Vision, Natural Language Processing, and Generative AI. 

The curriculum is specifically calibrated for mid-to-senior ML engineering roles, prioritizing bare-metal mathematics, complex network topologies (Keras Functional API), custom training loops, and advanced data pipelines.

---

## 🗺️ The 4-Project Architectural Roadmap

### 📊 Lab 1: The Bare-Metal Perceptron & The Great Keras Race
* **Objective:** Strip away framework magic. Build a multi-layer perceptron from absolute scratch using only vector math in NumPy, then implement its exact architectural twin in TensorFlow/Keras to benchmark convergence rates and structural constraints.
* **Core Mechanisms Covered:** * Vectorized multi-layer backpropagation and matrix calculus.
  * Mathematical weight initializations (He/Kaiming vs. Xavier/Glorot).
  * Receptive gradients & debugging structural failure points (e.g., the Dying ReLU problem).
* **Interview Defense Vectors:** Vectorization dimensions, tracking gradient shape anomalies, optimization trajectories (Batch Gradient Descent vs. SGD with momentum).

### 👁️ Lab 2: Dual-Mode Semantic Segmentation (Custom CNN vs. Transfer Learning U-Net)
* **Objective:** Move beyond naive image classification into spatial pixel-level scene parsing. Construct a U-Net architecture twice: first by engineering a custom deep CNN encoder from scratch to master spatial math, and second by swapping it out for a pre-trained feature extractor (e.g., MobileNetV2/ResNet50).
* **Core Mechanisms Covered:**
  * Spatial dimension transformations ($W_{out} = \lfloor \frac{W_{in} - K + 2P}{S} \rfloor + 1$).
  * Keras Functional API for multi-input/multi-output non-linear execution topologies.
  * Structural skip connections to recover spatial loss during encoding.
  * Non-standard loss functions (Dice Loss, Jaccard Index) vs. Pixel-wise Cross-Entropy.
* **Interview Defense Vectors:** Receptive field calculation, feature map compression, exploitation of frozen pre-trained layers, and resolution recovery mechanics.

### 📝 Lab 3: Sequence Dynamics & Linguistic Analysis (1D-CNN vs. LSTM/GRU)
* **Objective:** Master temporal dependencies and structural text embeddings. Build a comparative natural language pipeline evaluating the spatial feature-extraction capacity of 1D Convolutional networks against recurrent sequence models (LSTM/GRU).
* **Core Mechanisms Covered:**
  * Clean text vectorization, vocabulary indexing, and dense token embedding layer design.
  * Recurrent gate mathematics (Forget, Input, Output gates) and hidden state transitions.
  * Structural vulnerabilities in long-range dependencies (Vanishing/Exploding gradients).
* **Interview Defense Vectors:** Mathematical proofs of vanishing gradients in Vanilla RNNs, structural runtime speedups of 1D-CNNs over LSTMs, and proper padding selection (`pre` vs. `post`) for recurrent states.

### 🌌 Lab 4: Generative Latent Spaces via Variational Autoencoders (VAEs)
* **Objective:** Transition from discriminative neural architectures to generative probabilistic modeling. Build a Variational Autoencoder that maps raw inputs onto a continuous latent space distribution, allowing for arbitrary high-fidelity data generation.
* **Core Mechanisms Covered:**
  * Autoencoder structural bottleneck properties.
  * Probabilistic framework optimization: Reconstruction loss combined with Kullback-Leibler (KL) Divergence.
  * The Reparameterization Trick ($z = \mu + \sigma \odot \epsilon$) to sustain backprop path continuity.
  * Deep Keras customization: Overriding `Model.train_step` for non-standard training logic.
* **Interview Defense Vectors:** The exact mathematical need for the reparameterization trick, balancing latent reconstruction vs. spatial regularization, and the fundamental differences between VAEs and GANs.

---

## 🎓 Target Interview Readiness Profile

By the conclusion of this refresher track, the codebase demonstrates fluency across:

1.  **Low-Level Theory:** Deriving backpropagation and loss formulations by hand.
    
2.  **Framework Mastery:** Confidently leveraging the `tf.keras` Functional API, writing custom training steps, and constructing efficient `tf.data.Dataset` pipelines.
    
3.  **Architectural Pragmatism:** Knowing exactly when to build from scratch versus when to use transfer learning to optimize a business compute budget.

---

### 📝 AI Collaboration & Authenticity Disclosure

> **Note on Project Development:** The architectural blueprints, curriculum syllabus, and engineering constraints outlined in this repository were designed in collaboration with Google Gemini to mimic rigorous, university-level advanced laboratory courses (ML2).
> 
> While AI was leveraged as an algorithmic co-pilot for high-level structuring, curriculum design, and interview-defense calibration, **all source code, mathematical derivations, matrix vectorizations, and performance optimizations are executed, coded, and debugged entirely by me from scratch.** This repository serves as a transparent testament to my hands-on technical capabilities and core engineering intuition.
