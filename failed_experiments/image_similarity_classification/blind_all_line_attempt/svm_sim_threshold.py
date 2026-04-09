"""
Training linear SVM on similarity scores to obtain
decision boundary as threshold
"""

# Imports
print("Importing packages...")
from pathlib import Path
import numpy as np
from sklearn import svm
import time
print("Imports done")

# Constants (settings, paths etc)
NPY_FOLDER_PATH = (
    Path(__file__).parent / "embeddings_etc"
)
POS_TRUTH_NPY_NAME = "all_pos_sim_scores" # with or without .npy
NEG_TRUTH_NPY_NAME = "all_neg_sim_scores"

# Script

print("Opening files at", NPY_FOLDER_PATH)
pos_path = (NPY_FOLDER_PATH / POS_TRUTH_NPY_NAME).with_suffix(".npy")
neg_path = (NPY_FOLDER_PATH / NEG_TRUTH_NPY_NAME).with_suffix(".npy")
pos_arr = np.load(pos_path)
neg_arr = np.load(neg_path)
print(".npy files opened")

pos_n = len(pos_arr)
neg_n = len(neg_arr)

X = np.hstack((pos_arr, neg_arr)).reshape(-1, 1)
y = np.hstack((np.ones(pos_n), np.zeros(neg_n)))

# I think the default hyperparameters are fine?
lin_svm = svm.LinearSVC(random_state=2025)
# I read the docs, apparently there's a little randomness?

print("Training SVM...")
start_time = time.time()
lin_svm.fit(X, y)
end_time = time.time()
train_time = end_time - start_time
print("Training complete in", train_time, "seconds")

coef = lin_svm.coef_
intercept = lin_svm.intercept_
print("coef:", coef)
print("intercept:", intercept)

print("Predict first positive sample:")
print(pos_arr[0])
print("Prediction:", lin_svm.predict(np.array(pos_arr[0]).reshape(1, -1)))
print("Predict first negative sample:")
print(neg_arr[0])
print("Prediction:", lin_svm.predict(np.array(neg_arr[0]).reshape(1, -1)))

print("Other predictions:")
test_n = 50
for x in range(1, test_n):
    print("at", x/test_n, "=", lin_svm.predict(np.array(x/test_n).reshape(1, -1)))

print("Positive samples below coef:", len(tuple(x for x in pos_arr if x < coef)))
print("Positive samples at least coef:", len(tuple(x for x in pos_arr if x >= coef)))
print("Negative samples below coef:", len(tuple(x for x in neg_arr if x < coef)))
print("Negative samples at least coef:", len(tuple(x for x in neg_arr if x >= coef)))
