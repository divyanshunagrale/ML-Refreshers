# Machine Learning & Applied ML Architecture Refresher (ML1)

This repository serves as a structured, hands-on refresher covering the end-to-end machine learning lifecycle, from data exploration and preprocessing to model evaluation, feature engineering, and recommendation algorithms.

The curriculum progressively rebuilds core machine learning skills through four practical labs of increasing complexity, emphasizing scikit-learn pipelines, rigorous evaluation methodologies, leakage prevention, and sound experimental practices.

---

> **🚧 Development Status:** Work on this refresher track is temporarily on hold while I prioritize the **Deep Learning (ML2)** track. This repository will be revisited to complete the planned labs once that focus shifts back to core machine learning.

---

## 🗺️ The 4-Project Applied ML Roadmap

### 🚢 Lab 1: The Titanic Challenge — Foundations of Predictive Modeling

* **Objective:** Re-establish the fundamentals of supervised learning by building a complete classification pipeline to predict passenger survival. Focus on transforming raw tabular data into reliable model inputs and establishing meaningful baseline performance.
* **Core Mechanisms Covered:**
  * Exploratory Data Analysis (EDA), missing-value imputation, and categorical feature encoding.
  * `OneHotEncoder`, `Pipeline`, and `ColumnTransformer` for reproducible preprocessing.
  * Baseline model implementation using Logistic Regression and Random Forests.
  * Classification evaluation using accuracy, ROC-AUC, and comparative model analysis.
* **Interview Defense Vectors:** Data leakage during preprocessing, the role of train/test splits, the difference between accuracy and ROC-AUC, and the advantages of encapsulating preprocessing within a pipeline.
* **Dataset:** [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

### ❤️ Lab 2: Heart Disease Prediction — Evaluation Beyond Accuracy

* **Objective:** Advance from basic classification to more rigorous model evaluation using a clinical dataset. Investigate how class distributions, feature scaling, and decision thresholds affect predictive performance.
* **Core Mechanisms Covered:**
  * Data cleaning, missing-value handling, and class distribution analysis.
  * Scaling numerical features within pipelines to prevent data leakage.
  * Confusion matrices, precision, recall, F1-score, and ROC curves.
  * Feature importance and Logistic Regression coefficients for model interpretation.
  * Class imbalance analysis and appropriate evaluation metric selection.
* **Interview Defense Vectors:** Precision versus recall trade-offs, false positives versus false negatives, threshold tuning, the limitations of accuracy on imbalanced datasets, and interpreting feature importance without confusing correlation with causation.
* **Dataset:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

### 📈 Lab 3: Real-World Feature Engineering & Validation

* **Objective:** Bridge the gap between introductory datasets and realistic machine learning problems. Develop the ability to engineer useful features, identify data leakage, and choose validation strategies that reflect how a model will encounter data in practice.

* **Core Mechanisms Covered:**

  * **Option A — Bank Marketing Classification:**
    * Handling categorical variables with high cardinality.
    * Precision-Recall curves and evaluation under class imbalance.
    * Framing predictions around business objectives and campaign outcomes.
    * Feature importance, threshold selection, and leakage investigation.

  * **Option B — NYC Taxi Trip Duration Regression:**
    * Feature engineering from pickup/drop-off coordinates and timestamps.
    * Outlier detection, handling anomalous records, and log transformations.
    * Regression evaluation using MAE, RMSE, and appropriate target transformations.
    * Time-aware validation and leakage prevention where the data's temporal structure requires it.

* **Interview Defense Vectors:** Identifying target leakage, distinguishing random from time-based validation, handling high-cardinality categorical features, justifying feature transformations, and translating model metrics into practical outcomes.
* **Datasets:**
  * [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
  * [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)

### 🎬 Lab 4: MovieLens — Collaborative Filtering & Recommendation Systems

* **Objective:** Progress beyond conventional supervised learning into personalized recommendation systems. Build a recommender using user-item rating interactions, evaluate its ability to predict preferences, and explore how to rank relevant items for individual users.
* **Core Mechanisms Covered:**
  * Understanding user-item interaction matrices, sparsity, and explicit versus implicit feedback.
  * Implementing collaborative filtering with the Surprise library.
  * Establishing baseline recommenders and comparing collaborative filtering approaches.
  * Chronological train/test splitting to simulate future recommendations.
  * Rating prediction with RMSE and recommendation ranking with Recall@K.
  * Generating Top-N recommendations while accounting for items users have already rated.
* **Interview Defense Vectors:** User-based versus item-based collaborative filtering, the cold-start problem, matrix sparsity, rating prediction versus ranking quality, temporal evaluation, and the limitations of random train/test splits for recommendation systems.
* **Dataset:** [MovieLens 100K — GroupLens](https://grouplens.org/datasets/movielens/100k/)

---

## 🎓 Target Interview Readiness Profile

By the conclusion of this refresher track, the codebase aims to demonstrate practical competence across:

1. **Data Preparation & Feature Engineering:** Cleaning tabular datasets, handling missing values, encoding categorical variables, scaling numerical features, and engineering domain-specific predictors.

2. **Reliable ML Evaluation:** Selecting appropriate metrics, analyzing classification and regression errors, evaluating imbalanced datasets, and designing validation strategies that reduce leakage and reflect real-world use.

3. **Scikit-learn Engineering:** Building reusable preprocessing and modeling pipelines, applying sound experimental practices, comparing baseline models against more complex alternatives, and maintaining consistent transformations between training and inference.

4. **Model Interpretation & Practical Reasoning:** Explaining model behavior, understanding performance trade-offs, and connecting evaluation metrics to the problem being solved.

5. **Recommendation Systems:** Understanding user-item interactions, implementing collaborative filtering, evaluating recommendation quality, and distinguishing rating prediction from personalized ranking.

---

### 📝 AI Collaboration & Authenticity Disclosure

> **Note on Project Development:** The lab sequence, curriculum structure, and learning objectives were developed with AI assistance to provide a progressively challenging, project-based machine learning refresher.
>
> The repository is intended to document hands-on implementation, experimentation, debugging, and technical learning. Model results, methodological choices, and implementation claims will be documented based on the work actually completed, rather than assumed from the proposed curriculum.
