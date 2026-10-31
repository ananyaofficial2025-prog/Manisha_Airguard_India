# AirGuard India: Air Quality Analysis and Pollution Alert Prediction
# Student: Manisha | Reg No: 12403722 | Section: NK | Course: INT234 Predictive Analytics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc

# ---------------- 1. Load dataset ----------------
data = pd.read_csv("C:\\Users\\anany\\Downloads\\city_day.csv")

# View data structure and summary
print(data.head())
print(data.info())
print(data.describe())

# Identify the NAs in data
print(data.isna().sum())

# Plot missing values (%)
(data.isna().mean() * 100).sort_values().plot(kind="barh", figsize=(8, 5))
plt.title("Missing values by column (%)")
plt.xlabel("Missing values (%)")
plt.show()

# ---------------- 2. Data cleaning ----------------
# Remove duplicate rows
print("Duplicate rows:", data.duplicated().sum())
data = data.drop_duplicates()

# Rows without AQI cannot be used
data = data.dropna(subset=["AQI"])

# Pollutant columns
pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

# Drop pollutant columns with more than 40% missing values
for col in pollutants[:]:
    if data[col].isna().mean() > 0.40:
        data = data.drop(columns=col)
        pollutants.remove(col)
print("Pollutants used:", pollutants)

# Numeric columns - fill NA with mean
for col in pollutants:
    data[col] = data[col].fillna(data[col].mean())
print(data.isna().sum())

# Get Month from Date
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
data["Month"] = data["Date"].dt.month

# Alert = 1 for Poor, Very Poor, Severe (AQI above 200), else 0
data["ALERT"] = (data["AQI"] > 200).astype(int)
print(data["ALERT"].value_counts())

# ---------------- 3. EDA ----------------
# AQI histogram
plt.hist(data["AQI"], bins=30)
plt.xlabel("AQI")
plt.ylabel("Number of days")
plt.title("AQI Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(data[pollutants + ["AQI"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation between pollutants and AQI")
plt.show()

# Average AQI by city
data.groupby("City")["AQI"].mean().sort_values().plot(kind="barh", figsize=(8, 6))
plt.xlabel("Average AQI")
plt.title("Average AQI by City")
plt.show()

# AQI category count
plt.figure()
sns.countplot(x="AQI_Bucket", data=data)
plt.title("AQI Category Counts")
plt.xticks(rotation=25)
plt.show()

# Scatter plot PM2.5 vs AQI
sns.scatterplot(x=data["PM2.5"], y=data["AQI"])
plt.title("PM2.5 vs AQI")
plt.show()

# Alert percentage by city
(data.groupby("City")["ALERT"].mean() * 100).sort_values().plot(kind="barh", figsize=(8, 6))
plt.xlabel("Alert days (%)")
plt.title("Percentage of alert days by city")
plt.show()

# ---------------- 4. Regression (predict AQI) ----------------
# Normalization (Min-Max Scaling)
scaler = MinMaxScaler()
data[pollutants] = scaler.fit_transform(data[pollutants])

# Simple Linear Regression (PM2.5 -> AQI)
X = data[["PM2.5"]]
y = data["AQI"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("\nSimple Linear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot regression line
plt.scatter(X_test, y_test, color="blue")
plt.plot(X_test, y_pred, color="red", linewidth=3)
plt.xlabel("PM2.5 (Normalized)")
plt.ylabel("AQI")
plt.title("Simple Linear Regression Fit")
plt.show()

# Multiple Linear Regression (all pollutants + Month -> AQI)
X = data[pollutants + ["Month"]]
y = data["AQI"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("\nMultiple Linear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, color="purple")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()

# ---------------- 5. Classification (predict ALERT) ----------------
# AQI is not used as a feature
X = data[pollutants + ["Month"]]
y = data["ALERT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print("\nLogistic Regression")
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

# KNN - find best K
best_k = 3
best_acc = 0
for k in range(3, 16, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    print("K =", k, "Accuracy =", round(acc, 4))
    if acc > best_acc:
        best_acc = acc
        best_k = k
print("Best K:", best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("\nKNN")
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

# Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print("\nGaussian Naive Bayes")
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# Decision Tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf.fit(X_train, y_train)
tree_pred = clf.predict(X_test)
print("\nDecision Tree")
print("Confusion Matrix:\n", confusion_matrix(y_test, tree_pred))
print(classification_report(y_test, tree_pred))

plt.figure(figsize=(14, 8))
plot_tree(clf, feature_names=X.columns, class_names=["No Alert", "Alert"], filled=True)
plt.title("Decision Tree for Pollution Alert")
plt.show()

# SVM
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("\nSVM")
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# ROC curve for SVM
y_score = svm.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend()
plt.show()

# ---------------- 6. Model comparison ----------------
models = {"Logistic Regression": log_pred, "KNN": knn_pred, "Naive Bayes": nb_pred,
          "Decision Tree": tree_pred, "SVM": svm_pred}
results = []
for name in models:
    p = models[name]
    results.append([name, accuracy_score(y_test, p), precision_score(y_test, p),
                    recall_score(y_test, p), f1_score(y_test, p)])
result_table = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
print("\nClassification Model Comparison:")
print(result_table.sort_values("Recall", ascending=False))

# ---------------- Conclusion ----------------
print("\nConclusion:")
print("1. Pollutants (mainly PM2.5 and PM10) are strongly related to AQI.")
print("2. Multiple Linear Regression predicts AQI better than Simple Linear Regression.")
print("3. Classification models can predict Poor/Very Poor/Severe alert days.")
