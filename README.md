# Iris Dataset ML Project

## 1) Task objective

The objective of this project is to build a complete machine learning pipeline on the **Iris dataset** to classify flower species based on their measurements.

The notebook focuses on:

- Understanding and exploring the dataset structure
- Visualizing feature distributions and relationships
- Preparing data for model training
- Training a classification model (SVM)
- Evaluating performance using multiple metrics
- Testing the model on a new sample flower

The target variable is **Species** (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) and the input features are:

- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`

---

## 2) My approach

I followed a step-by-step machine learning workflow in the notebook:

### A. Data loading and initial inspection
- Loaded `Iris.csv` using pandas
- Displayed top rows (`df.head()`)
- Checked:
  - shape: **150 rows × 6 columns**
  - column names
  - datatypes
  - descriptive statistics (`df.describe()`)
  - missing values (**none found**)

### B. Exploratory Data Analysis (EDA)
To understand patterns and separability among classes, I created:

1. **Scatter plots**
   - Sepal Length vs Sepal Width
   - Sepal Length vs Petal Length
   - Petal Length vs Petal Width

2. **Histograms**
   - Distribution of each numeric feature

3. **Box plots**
   - Spread and possible outliers in each numeric feature

I excluded `Id` from meaningful feature analysis because it is only an index-like identifier.

### C. Data preparation
- Defined features and target:
  - `X = df.drop(['Id', 'Species'], axis=1)`
  - `y = df['Species']`
- Performed train-test split:
  - **80% train / 20% test**
  - `random_state=42`
  - `stratify=y` to preserve class balance
- Applied feature scaling:
  - Used `StandardScaler`
  - Fit on training set and transform both train/test sets

### D. Model training
- Trained a **Support Vector Machine (SVC)** with:
  - `kernel='rbf'`
  - `random_state=42`

### E. Model evaluation
Evaluated predictions on the test set using:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion matrix
- Classification report
- 5-fold cross-validation on training data

### F. New sample prediction
- Predicted class for a custom flower:
  - Measurements: `[5.8, 3.2, 4.5, 1.2]`
- Output predicted species and decision-function confidence scores

---

## 3) Results and insights

### Final model performance (test set)
- **Accuracy:** 0.9667 (96.67%)
- **Precision (weighted):** 0.9697
- **Recall (weighted):** 0.9667
- **F1-score (weighted):** 0.9666

### Confusion matrix
\[
\begin{bmatrix}
10 & 0 & 0 \\
0 & 9 & 1 \\
0 & 0 & 10
\end{bmatrix}
\]

This indicates:
- Perfect predictions for **Iris-setosa** and **Iris-virginica** in this split
- One **Iris-versicolor** sample misclassified as **Iris-virginica**

### Classification report insight
- `Iris-setosa` is perfectly separable in this run
- Most confusion occurs between `Iris-versicolor` and `Iris-virginica`, which is expected because their feature ranges overlap more than setosa

### Cross-validation insight
- 5-fold CV scores: `[0.9167, 1.0000, 0.9583, 0.9583, 1.0000]`
- **Mean CV accuracy:** 0.9667
- **Std dev:** 0.0312

This suggests the model is stable and generalizes well on this dataset.

### Practical sample prediction
For input `[5.8, 3.2, 4.5, 1.2]`, the model predicted:

- **Iris-versicolor**

which is a realistic and expected classification for those measurements.

---

## Conclusion

This project successfully demonstrates a full machine learning classification pipeline on the Iris dataset—from EDA to deployment-style single prediction.  
The SVM model achieved strong and consistent performance, showing that even a relatively simple setup can produce highly accurate multiclass classification on clean, structured data.
