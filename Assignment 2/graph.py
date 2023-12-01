import numpy as np
import matplotlib.pyplot as plt

# True class labels
y = np.array(['-', '+', '-', '-', '+', '-', '+', '+', '-', '+'])

# Probabilities from classifiers C1 and C2
proba_c1 = np.array([0.1, 0.15, 0.2, 0.3, 0.31, 0.4, 0.62, 0.77, 0.81, 0.95])
proba_c2 = np.array([0.25, 0.49, 0.05, 0.35, 0.66, 0.6, 0.7, 0.65, 0.55, 0.99])

# Initialize lists to store TPR and FPR for each classifier
tpr_c1 = []
fpr_c1 = []
tpr_c2 = []
fpr_c2 = []

# Define a range of thresholds
thresholds = np.linspace(0, 1, 100)

# Calculate TPR and FPR for each threshold for both classifiers
for threshold in thresholds:
    # For Classifier C1
    predicted_labels_c1 = np.where(proba_c1 >= threshold, '+', '-')
    tpr_c1.append(np.sum((y == '+') & (predicted_labels_c1 == '+')) / np.sum(y == '+'))
    fpr_c1.append(np.sum((y == '-') & (predicted_labels_c1 == '+')) / np.sum(y == '-'))

    # For Classifier C2
    predicted_labels_c2 = np.where(proba_c2 >= threshold, '+', '-')
    tpr_c2.append(np.sum((y == '+') & (predicted_labels_c2 == '+')) / np.sum(y == '+'))
    fpr_c2.append(np.sum((y == '-') & (predicted_labels_c2 == '+')) / np.sum(y == '-'))

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_c1, tpr_c1, label='C1')
plt.plot(fpr_c2, tpr_c2, label='C2')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.grid(True)
plt.show()
