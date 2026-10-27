# Heart Disease Prediction

This *"Lab"* assignment is meant for me to start getting hands-on again with Machine Learning.

In this *"assignment"* we work with the Heart Disease UCI dataset.

We use a clean version of the dataset. Link to the data can be found [here](https://gist.github.com/notreallyme2/8d20f6b5bcc3606e0541cbdf2ee0a7a6).

---

## Run this Yourself

To run the notebooks yourself, the only prerequisites, outside the obvious having access to an environment to the jupyter notebooks, is to have `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn` installed.

---

## Model Metrics

In the modelling notebook, we have trained the following models along with their metrics:

| **Model**             | **Precision**  | **Recall**    | **F1**    | **ROC-AUC**   |
| --------------------- | ------------- | ------------- | --------- | ------------- |
| Logistic Regression   | 0.89          | 0.89          | 0.89      | 0.95          |
| SVC                   | 0.89          | 0.89          | 0.89      | 0.93          |
| Random Forest         | 0.89          | 0.89          | 0.89      | 0.94          |

Since the logistic regression model had the better ROC-AUC score, we did some threshold tuning to improve that model.

For that, we plotted a Precision-Recall curve and pick the decision threshold the maximized the F1 score. This improved the model to have better precision of `0.91`and better recall and f1 scores of `0.9`. The optimal threshold was found to be `0.109`, which brought down the number of false negatives (which is the most important metric to be minimized for medical models) from 5 down to just 1.

---

## What I Learned

- The `ROC-AUC` score was just another metric while doing the last assignment, but this assignment I really built a genuine understanding what it is and why it is important.
- Since this was my first time working with a medical dataset and modelling a medical prediction model, concepts like the `PR curve` and `threshold tuning` were new to me, even with a MSc in Data Science, which I wouldn't have learned about without this project. I came to understand and appreciate the care and sensitivity that goes into building a predictive medical model because of how a bad prediction can be a matter of life and death.
- With all these new concepts, this one took me a lot of time. I became a bit lazier and procrastinated near the end of this assignment. But I've come to appreciate the fact the learning is supposed to be hard. In a world where vibe coding seems to be the answer for many new developers and CS students, I feel this extra bit of effort that I'm putting in to relearn ML will be a lot more valuable in my career.

---

Date completed: `28-03-2025`