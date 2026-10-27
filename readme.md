# ML Refreshers — Machine Learning & Deep Learning

This repository documents a structured, hands-on refresher in Machine Learning (ML) and Deep Learning (DL), progressing from foundational mathematical concepts and classical ML techniques to modern neural network architectures and advanced learning paradigms.

The goal is to rebuild core theoretical understanding through practical implementation, reinforce engineering best practices, and develop the ability to explain and defend machine learning design decisions in technical interviews.

Rather than relying solely on high-level abstractions, the curriculum emphasizes understanding how algorithms work, why architectural choices matter, and how models should be trained, evaluated, and validated.

----------

## 🗺️ Repository Roadmap

The repository is organized into two complementary tracks, each containing four practical labs of increasing complexity.

### 📊 ML1: Machine Learning & Applied ML

**Focus:** Classical machine learning, data preparation, feature engineering, model evaluation, and recommendation systems.

This track revisits the end-to-end ML workflow, starting with introductory classification problems and progressing toward more realistic datasets and collaborative filtering.

**Core areas covered:**

- Exploratory Data Analysis (EDA) and data preprocessing.

- Missing-value imputation, categorical encoding, and feature scaling.

- Scikit-learn pipelines and reproducible experimentation.

- Classification and regression metrics.

- Feature engineering, data leakage prevention, and validation strategies.

- Collaborative filtering and recommendation ranking.

**Lab progression:**

1. Titanic — Foundations of Predictive Modeling.

2. Heart Disease — Evaluation Beyond Accuracy.

3. Bank Marketing or NYC Taxi Trip Duration — Feature Engineering & Validation.

4. MovieLens — Collaborative Filtering & Recommendation Systems.

Explore the [Machine Learning track](./Machine%20Learning/readme.md) for the detailed curriculum.

### 🧠 ML2: Deep Learning & Neural Network Architectures

**Focus:** Neural network fundamentals, mathematical implementation, architectural design, and advanced deep learning workflows.

This track moves from implementing neural networks using low-level numerical operations to constructing architectures for computer vision, sequence modeling, and generative AI.

**Core areas covered:**

- Forward propagation, backpropagation, and matrix calculus.

- Weight initialization, gradient behavior, and optimization.

- Convolutional neural networks and semantic segmentation.

- Sequence modeling with 1D-CNNs, LSTMs, and GRUs.

- Variational Autoencoders (VAEs) and probabilistic latent spaces.

- TensorFlow/Keras Functional API, custom training steps, and data pipelines.

**Lab progression:**

1. The Bare-Metal Perceptron & The Great Keras Race.

2. Dual-Mode Semantic Segmentation — Custom CNN vs. Transfer Learning U-Net.

3. Sequence Dynamics & Linguistic Analysis — 1D-CNN vs. LSTM/GRU.

4. Generative Latent Spaces via Variational Autoencoders.

Explore the [Deep Learning track](./Deep%20Learning/readme.md) for the detailed curriculum.

----------

## 🧭 Learning Philosophy

The curriculum follows a few guiding principles:

1. **Understand before abstracting:** Develop an understanding of the mathematics and underlying mechanisms rather than treating ML frameworks as black boxes.

2. **Build incrementally:** Start with manageable problems and progressively introduce more complex datasets, architectures, and training strategies.

3. **Evaluate rigorously:** Choose metrics appropriate to the task, investigate failure cases, and avoid misleading results caused by data leakage or unsuitable validation strategies.

4. **Prioritize reproducibility:** Use structured preprocessing pipelines, consistent experiments, and documented implementation decisions.

5. **Defend technical decisions:** Be able to explain not just what was implemented, but why a particular algorithm, architecture, loss function, or evaluation strategy was chosen.

6. **Balance theory with practical engineering:** Connect mathematical understanding to working implementations, computational constraints, and real-world use cases.

----------

## 🎓 Target Outcomes

By completing both tracks, the intended outcomes are:

- **Mathematical Foundations:** A stronger understanding of classical ML algorithms, neural network optimization, gradient propagation, and loss functions.

- **Applied ML Competence:** The ability to prepare datasets, engineer features, build reliable pipelines, and evaluate predictive models.

- **Deep Learning Competence:** Practical experience implementing and comparing neural network architectures across multiple problem domains.

- **Evaluation & Debugging:** The ability to diagnose model failures, reason about performance metrics, and identify common implementation and validation mistakes.

- **Engineering Discipline:** Familiarity with reproducible experiments, modular implementations, and framework features that support maintainable ML workflows.

- **Technical Interview Readiness:** The ability to communicate core concepts, explain trade-offs, and justify engineering decisions with mathematical and practical reasoning.

----------

## 📁 Repository Structure

```text
ML Refreshers/
├── Deep Learning/
│   ├── Lab_1/
│   ├── Lab_2/
│   ├── Lab_3/
│   ├── Lab_4/
│   └── readme.md
├── Machine Learning/
│   ├── Lab_1/
│   ├── Lab_2/
│   ├── Lab_3/
│   ├── Lab_4/
│   └── readme.md
└── readme.md
```

Each track has its own README outlining its objectives, lab progression, key technical concepts, and target interview topics. Individual lab directories contain the corresponding implementations, experiments, and supporting documentation.

----------

## 📝 AI Collaboration & Authenticity Disclosure

> **Note on Project Development:** The curriculum structure, lab blueprints, and learning objectives were developed with AI assistance to establish a progressively challenging, project-based refresher.
>
> The repository is intended to document hands-on implementation, experimentation, debugging, and technical learning. Claims about completed work, model performance, and implementation decisions will reflect the work actually performed and documented.
