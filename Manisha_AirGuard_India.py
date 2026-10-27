# Project by Manisha
# %% [markdown]
# This first cell stores student details and finds the project folder. The same
# script can run from Spyder or from a command prompt.
# %%
from pathlib import Path
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, log_loss,
    roc_auc_score, roc_curve, confusion_matrix, classification_report)
import statsmodels.api as sm

STUDENT_NAME = "Manisha"
ROLL_NO = "10"
SECTION = "NK"
TEACHER = "Dr. Mrinalini Rana"
UNIVERSITY = "LOVELY PROFESSIONAL UNIVERSITY"
RANDOM_STATE = 42

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_PATH = BASE_DIR / "data" / "city_day.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")

# %% [markdown]
# This title cell prints the project and its student details for the console.
# %%
print("=" * 78)
print("AirGuard India: Air Quality Analysis and Pollution Alert Prediction")
print(f"Project by {STUDENT_NAME} | Roll No: {ROLL_NO} | Section: {SECTION}")
print(f"Course: INT234 Predictive Analytics (Python) | Teacher: {TEACHER}")
print(f"University: {UNIVERSITY} | RANDOM_STATE: {RANDOM_STATE}")
print("=" * 78)

# %% [markdown]
# Section 1 explains the problem. Citizens, schools, hospitals, and government
# or environment agencies need a clear warning when air becomes harmful.
# %%
print("SECTION 1: Introduction and Problem Statement")
print("AirGuard studies Indian city pollutant readings, estimates AQI, and raises")
print("an ALERT for Poor, Very Poor, or Severe air. It is intended to support")
print("citizens, schools, hospitals, and government/environment agencies.")

# %% [markdown]
# Section 2 starts by loading the provided local CSV only. The printed column
# list helps adapt the work if the file uses slightly different names.
# %%
print("\nSECTION 2: Unit I - Data Preparation and EDA")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Place the required local dataset at: {DATA_PATH}")
raw = pd.read_csv(DATA_PATH)
print("Dataset columns:", list(raw.columns))
print("Dataset shape:", raw.shape)
print("First five rows:\n", raw.head().to_string(index=False))
print("Dataset information:")
raw.info()
print("Descriptive statistics:\n", raw.describe(include="all").to_string())

# %% [markdown]
# The missing-value table and chart show where the local data is incomplete.
# Very incomplete pollutant columns are removed, and missing AQI rows are
# removed because the target must never be imputed.
# %%
missing_pct = (raw.isna().mean() * 100).sort_values(ascending=False)
missing_table = missing_pct.rename("Missing_Percent").reset_index().rename(columns={"index": "Column"})
print("Missing value percentage table:\n", missing_table.to_string(index=False))
fig, ax = plt.subplots(figsize=(11, 5))
missing_pct.sort_values().plot(kind="barh", ax=ax, color="#4C9F70")
ax.set_title("Missing values by column (%)")
ax.set_xlabel("Missing (%)")
ax.set_ylabel("Column")
fig.savefig(OUTPUT_DIR / "missing_values.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

# %% [markdown]
# Columns with more than 40% missing observations are excluded (often Xylene).
# AQI-missing rows are dropped. Remaining pollutant gaps use each city's
# median, with the overall median as a fallback.
# %%
df = raw.copy()
drop_cols = []
for col in df.columns:
    if col not in ["City", "Date", "AQI", "AQI_Bucket"]:
        if df[col].isna().mean() > 0.40:
            drop_cols.append(col)
print("Dropped very incomplete pollutant columns:", drop_cols)
df = df.drop(columns=drop_cols)
df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
df = df.dropna(subset=["AQI"]).copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["Date", "City"])
df["Year"] = df["Date"].dt.year.astype(int)
df["Month"] = df["Date"].dt.month.astype(int)
pollutants = []
possible_pollutants = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]
for col in possible_pollutants:
    if col in df.columns:
        pollutants.append(col)
for col in pollutants:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    city_medians = df.groupby("City")[col].transform("median")
    df[col] = df[col].fillna(city_medians).fillna(df[col].median())
    if df[col].isna().any():
        df[col] = df[col].fillna(0)
bucket = df["AQI_Bucket"].astype("string").str.strip().str.title()
df["AQI_Bucket_Clean"] = bucket
df.loc[df["AQI_Bucket_Clean"].isna(), "AQI_Bucket_Clean"] = np.where(
    df.loc[df["AQI_Bucket_Clean"].isna(), "AQI"] > 200, "Poor", "Moderate")
df["ALERT"] = df["AQI_Bucket_Clean"].isin(["Poor", "Very Poor", "Severe"]).astype(int)
city_encoder = LabelEncoder()
df["City_Encoded"] = city_encoder.fit_transform(df["City"].astype(str))
print("Cleaned shape:", df.shape, "| rows with valid AQI retained:", len(df))
print("ALERT class distribution (0=acceptable/moderate, 1=Poor or worse):\n", df["ALERT"].value_counts().sort_index().to_string())
print("Alert percentage:", round(df["ALERT"].mean() * 100, 2), "%")

# %% [markdown]
# Outliers are capped at the 1.5-IQR fences for pollutant predictors; no rows
# are removed. This can tone down genuine pollution spikes, so the boxplots
# show the change and AQI target values remain untouched.
# %%
plot_pollutants = pollutants.copy()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
df[plot_pollutants].boxplot(ax=axes[0], rot=60)
axes[0].set_title("Pollutants before IQR capping")
for col in plot_pollutants:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
df[plot_pollutants].boxplot(ax=axes[1], rot=60)
axes[1].set_title("Pollutants after IQR capping")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "iqr_before_after.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

# %% [markdown]
# These exploratory charts summarize AQI, its association with pollutants, and
# the differences among cities and AQI categories.
# %%
fig, ax = plt.subplots(figsize=(9, 5))
df["AQI"].hist(bins=35, ax=ax, color="#377EB8")
ax.set_title("AQI distribution")
ax.set_xlabel("AQI")
ax.set_ylabel("Days")
fig.savefig(OUTPUT_DIR / "aqi_histogram.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

numeric_cols = pollutants + ["AQI"]
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
ax.set_title("Pollutant and AQI correlation heatmap")
fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

city_aqi = df.groupby("City")["AQI"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
city_aqi.sort_values().plot(kind="barh", ax=ax, color="#E07A5F")
ax.set_title("Average AQI by city")
ax.set_xlabel("Average AQI")
fig.savefig(OUTPUT_DIR / "city_average_aqi.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
order = df["AQI_Bucket_Clean"].value_counts().index
sns.countplot(data=df, x="AQI_Bucket_Clean", order=order, ax=ax, color="#81B29A")
ax.set_title("AQI category counts")
ax.set_xlabel("AQI category")
ax.set_ylabel("Days")
ax.tick_params(axis="x", rotation=25)
fig.savefig(OUTPUT_DIR / "aqi_bucket_counts.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df.sample(min(len(df), 5000), random_state=RANDOM_STATE), x="PM2.5", y="AQI", alpha=.45, ax=ax)
ax.set_title("PM2.5 compared with AQI")
fig.savefig(OUTPUT_DIR / "pm25_vs_aqi.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

pollutant_corr = df[pollutants + ["AQI"]].corr()["AQI"].drop("AQI").sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
pollutant_corr.plot(kind="barh", ax=ax, color="#577590")
ax.set_title("Pollutant-wise correlation with AQI")
ax.set_xlabel("Correlation")
fig.savefig(OUTPUT_DIR / "pollutant_aqi_correlation.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)
print("Pollutants with largest absolute correlation to AQI:\n", pollutant_corr.reindex(pollutant_corr.abs().sort_values(ascending=False).index).head(5).to_string())

# %% [markdown]
# These are descriptive pollution trends only. They summarize historical
# averages and do not forecast future AQI.
# %%
year_aqi = df.groupby("Year")["AQI"].mean()
fig, ax = plt.subplots(figsize=(9, 5))
year_aqi.plot(marker="o", ax=ax)
ax.set_title("Yearly average AQI (descriptive)")
ax.set_ylabel("Average AQI")
fig.savefig(OUTPUT_DIR / "yearly_average_aqi.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)
month_aqi = df.groupby("Month")["AQI"].mean()
fig, ax = plt.subplots(figsize=(9, 5))
month_aqi.reindex(range(1,13)).plot(marker="o", ax=ax)
ax.set_title("Monthly average AQI (descriptive seasonality)")
ax.set_xlabel("Month")
ax.set_ylabel("Average AQI")
fig.savefig(OUTPUT_DIR / "monthly_average_aqi.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)
top5 = city_aqi.head(5).index
year_city = df[df["City"].isin(top5)].groupby(["Year", "City"])["AQI"].mean().unstack()
fig, ax = plt.subplots(figsize=(10, 6))
year_city.plot(marker="o", ax=ax)
ax.set_title("Yearly AQI trend for five most polluted cities")
ax.set_ylabel("Average AQI")
fig.savefig(OUTPUT_DIR / "top5_city_yearly_trend.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)
fig, ax = plt.subplots(figsize=(12, 6))
city_aqi.sort_values().plot(kind="bar", ax=ax, color="#F2CC8F")
ax.set_title("City-wise average AQI comparison")
ax.tick_params(axis="x", rotation=75)
fig.savefig(OUTPUT_DIR / "city_comparison.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)
alert_pct_city = (df.groupby("City")["ALERT"].mean() * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
alert_pct_city.sort_values().plot(kind="barh", ax=ax, color="#D1495B")
ax.set_title("Percentage of alert days by city")
ax.set_xlabel("Alert days (%)")
fig.savefig(OUTPUT_DIR / "alert_percentage_by_city.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)
alert_pct_city.rename("Alert_Percent").to_csv(OUTPUT_DIR / "alert_percentages_by_city.csv")

# %% [markdown]
# Section 3 uses an 80/20 split for AQI regression. Scalers and polynomial
# transformations are fitted using training data only to prevent leakage.
# %%
print("\nSECTION 3: Unit II - Regression")
reg_features = pollutants + ["Year", "Month", "City_Encoded"]
X_reg = df[reg_features].astype(float)
y_reg = df["AQI"].astype(float)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.20, random_state=RANDOM_STATE)
reg_results = []
reg_models = {}

def add_regression_result(name, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    reg_results.append({"Model": name, "MAE": mean_absolute_error(actual, predicted), "MSE": mse,
                        "RMSE": np.sqrt(mse), "R2": r2_score(actual, predicted)})

simple = LinearRegression().fit(Xr_train[["PM2.5"]], yr_train)
simple_pred = simple.predict(Xr_test[["PM2.5"]])
add_regression_result("Simple Linear Regression", yr_test, simple_pred)
reg_models["Simple Linear Regression"] = (simple, ["PM2.5"])
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Xr_test["PM2.5"], yr_test, alpha=.35, label="Actual")
sort_idx = np.argsort(Xr_test["PM2.5"].values)
ax.plot(Xr_test["PM2.5"].values[sort_idx], simple_pred[sort_idx], color="red", label="Regression line")
ax.set_title("Simple linear regression: PM2.5 to AQI")
ax.set_xlabel("PM2.5")
ax.set_ylabel("AQI")
ax.legend()
fig.savefig(OUTPUT_DIR / "simple_regression_line.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

multiple = LinearRegression().fit(Xr_train, yr_train)
multiple_pred = multiple.predict(Xr_test)
add_regression_result("Multiple Linear Regression", yr_test, multiple_pred)
reg_models["Multiple Linear Regression"] = (multiple, reg_features)
ols = sm.OLS(yr_train, sm.add_constant(Xr_train)).fit()
print("OLS summary (training data):\n", ols.summary())
print("OLS coefficients show estimated AQI change per feature unit while other included features are held fixed; p-values are associations in this dataset, not proof of cause.")

for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    xp_train = poly.fit_transform(Xr_train)
    xp_test = poly.transform(Xr_test)
    model = LinearRegression().fit(xp_train, yr_train)
    pred = model.predict(xp_test)
    name = f"Polynomial Regression Degree {degree}"
    add_regression_result(name, yr_test, pred)
    reg_models[name] = (model, (poly, reg_features))

# %% [markdown]
# Actual-versus-predicted and residual plots provide a visual check of the
# multiple regression errors. The score table reports four requested metrics.
# %%
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(yr_test, multiple_pred, alpha=.4)
lims=[min(yr_test.min(), multiple_pred.min()), max(yr_test.max(), multiple_pred.max())]
ax.plot(lims, lims, "r--")
ax.set_title("Multiple regression: actual vs predicted AQI")
ax.set_xlabel("Actual AQI")
ax.set_ylabel("Predicted AQI")
fig.savefig(OUTPUT_DIR / "regression_actual_predicted.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(multiple_pred, yr_test - multiple_pred, alpha=.4)
ax.axhline(0, color="red", linestyle="--")
ax.set_title("Multiple regression residuals")
ax.set_xlabel("Predicted AQI")
ax.set_ylabel("Residual (actual - predicted)")
fig.savefig(OUTPUT_DIR / "regression_residuals.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)
print("Regression comparison so far:\n")
print(pd.DataFrame(reg_results).to_string(
    index=False, float_format="{:.4f}".format
))

# %% [markdown]
# Section 4 predicts ALERT. AQI and AQI_Bucket are excluded from all features
# to prevent target leakage. A validation set selects KNN's k, so the test set
# stays untouched until final evaluation.
# %%
print("\nSECTION 4: Unit III - Classification")
class_features = pollutants + ["City_Encoded", "Month"]
X_cls = df[class_features].astype(float)
y_cls = df["ALERT"].astype(int)

# Keep 20% of the data aside as the final test set.
Xc_train_all, Xc_test, yc_train_all, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.20, random_state=RANDOM_STATE, stratify=y_cls
)

# Split the training portion again to select KNN's k using validation data.
Xc_train, Xc_valid, yc_train, yc_valid = train_test_split(
    Xc_train_all, yc_train_all, test_size=0.20,
    random_state=RANDOM_STATE, stratify=yc_train_all
)
validation_scaler = StandardScaler()
Xc_train_valid_scaled = validation_scaler.fit_transform(Xc_train)
Xc_valid_scaled = validation_scaler.transform(Xc_valid)

k_scores = []
for k in range(3, 16):
    knn_model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn_model.fit(Xc_train_valid_scaled, yc_train)
    validation_predictions = knn_model.predict(Xc_valid_scaled)
    validation_accuracy = accuracy_score(yc_valid, validation_predictions)
    k_scores.append({"k": k, "Accuracy": validation_accuracy})

k_table = pd.DataFrame(k_scores)
k_table = k_table.sort_values(["Accuracy", "k"], ascending=[False, True])
best_k = int(k_table.iloc[0]["k"])
print("KNN validation accuracies:\n", k_table.sort_values("k").to_string(index=False))
print("Selected k using validation data:", best_k)
k_table = k_table.sort_values("k").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_table["k"], k_table["Accuracy"], marker="o")
ax.set_title("KNN validation accuracy by k")
ax.set_xlabel("k")
ax.set_ylabel("Validation accuracy")
fig.savefig(OUTPUT_DIR / "knn_accuracy_by_k.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

# Fit the final scaler on all training rows, then use it for train and test.
class_scaler = StandardScaler()
Xc_train_all_scaled = class_scaler.fit_transform(Xc_train_all)
Xc_test_scaled = class_scaler.transform(Xc_test)

class_results = []
class_predictions = {}
class_probabilities = {}

def record_classification_result(name, file_name, predictions, probabilities):
    """Save one model's test scores and confusion-matrix chart."""
    result = {
        "Model": name,
        "Accuracy": accuracy_score(yc_test, predictions),
        "Precision": precision_score(yc_test, predictions, zero_division=0),
        "Recall": recall_score(yc_test, predictions, zero_division=0),
        "F1": f1_score(yc_test, predictions, zero_division=0),
        "Log_Loss": log_loss(yc_test, probabilities, labels=[0, 1]),
        "ROC_AUC": roc_auc_score(yc_test, probabilities)
    }
    class_results.append(result)
    class_predictions[name] = predictions
    class_probabilities[name] = probabilities

    matrix = confusion_matrix(yc_test, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion matrix: " + name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.savefig(OUTPUT_DIR / file_name, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    plt.close(fig)

# 1. Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
)
logistic_model.fit(Xc_train_all_scaled, yc_train_all)
logistic_predictions = logistic_model.predict(Xc_test_scaled)
logistic_probabilities = logistic_model.predict_proba(Xc_test_scaled)[:, 1]
record_classification_result(
    "Logistic Regression", "confusion_logistic_regression.png",
    logistic_predictions, logistic_probabilities
)

# 2. K-Nearest Neighbours with the k chosen on validation data
knn_model = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn_model.fit(Xc_train_all_scaled, yc_train_all)
knn_predictions = knn_model.predict(Xc_test_scaled)
knn_probabilities = knn_model.predict_proba(Xc_test_scaled)[:, 1]
knn_name = "KNN (k=" + str(best_k) + ")"
record_classification_result(
    knn_name, "confusion_knn__k_" + str(best_k) + ".png",
    knn_predictions, knn_probabilities
)

# 3. Gaussian Naive Bayes
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(Xc_train_all, yc_train_all)
naive_bayes_predictions = naive_bayes_model.predict(Xc_test)
naive_bayes_probabilities = naive_bayes_model.predict_proba(Xc_test)[:, 1]
record_classification_result(
    "Gaussian Naive Bayes", "confusion_gaussian_naive_bayes.png",
    naive_bayes_predictions, naive_bayes_probabilities
)

# 4. Entropy Decision Tree, limited to depth 4 for an understandable plot
tree_model = DecisionTreeClassifier(
    criterion="entropy", max_depth=4, class_weight="balanced",
    random_state=RANDOM_STATE
)
tree_model.fit(Xc_train_all, yc_train_all)
tree_predictions = tree_model.predict(Xc_test)
tree_probabilities = tree_model.predict_proba(Xc_test)[:, 1]
record_classification_result(
    "Decision Tree", "confusion_decision_tree.png",
    tree_predictions, tree_probabilities
)
fig, ax = plt.subplots(figsize=(18, 9))
plot_tree(
    tree_model, feature_names=class_features,
    class_names=["No Alert", "Alert"], filled=True,
    rounded=True, fontsize=7, max_depth=4, ax=ax
)
ax.set_title("Decision Tree classifier (maximum depth 4)")
fig.savefig(OUTPUT_DIR / "decision_tree_depth4.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

# 5. Linear Support Vector Machine
svm_model = SVC(
    kernel="linear", probability=True, class_weight="balanced",
    random_state=RANDOM_STATE
)
svm_model.fit(Xc_train_all_scaled, yc_train_all)
svm_predictions = svm_model.predict(Xc_test_scaled)
svm_probabilities = svm_model.predict_proba(Xc_test_scaled)[:, 1]
record_classification_result(
    "SVM", "confusion_svm.png", svm_predictions, svm_probabilities
)

# Draw one ROC curve for each classification model.
fig, ax = plt.subplots(figsize=(8, 6))
for name, probabilities in class_probabilities.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        yc_test, probabilities
    )
    area = roc_auc_score(yc_test, probabilities)
    ax.plot(false_positive_rate, true_positive_rate,
            label=name + " (AUC=" + format(area, ".3f") + ")")
ax.plot([0, 1], [0, 1], "k--")
ax.set_title("ROC curves: classification models")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(fontsize=8)
fig.savefig(OUTPUT_DIR / "classification_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

class_table = pd.DataFrame(class_results)
class_table = class_table.sort_values(
    ["Recall", "F1"], ascending=[False, False]
).reset_index(drop=True)
best_class_name = class_table.iloc[0]["Model"]
print("Classification comparison (alert recall, then F1):\n")
print(class_table.to_string(index=False, float_format="{:.4f}".format))
print("Best baseline classification report:", best_class_name)
print(classification_report(
    yc_test, class_predictions[best_class_name], zero_division=0
))

# %% [markdown]
# Section 5 introduces a simple feed-forward neural network. A neuron combines
# inputs, a hidden layer holds intermediate calculations, activation adds
# nonlinearity, an epoch is a training pass, and loss measures prediction error.
# %%
print("\nSECTION 5: Foundational Neural Network")
nn_classifier=MLPClassifier(hidden_layer_sizes=(64,32),activation="relu",max_iter=500,early_stopping=True,random_state=RANDOM_STATE)
nn_classifier.fit(Xc_train_all_scaled, yc_train_all)
nn_pred = nn_classifier.predict(Xc_test_scaled)
nn_prob = nn_classifier.predict_proba(Xc_test_scaled)[:, 1]
record_classification_result(
    "MLPClassifier (64, 32)", "confusion_mlpclassifier.png", nn_pred, nn_prob
)

# Add the neural network to the ROC plot as well.
fig, ax = plt.subplots(figsize=(8, 6))
for name, probabilities in class_probabilities.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        yc_test, probabilities
    )
    area = roc_auc_score(yc_test, probabilities)
    ax.plot(false_positive_rate, true_positive_rate,
            label=name + " (AUC=" + format(area, ".3f") + ")")
ax.plot([0, 1], [0, 1], "k--")
ax.set_title("ROC curves including MLPClassifier")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(fontsize=7)
fig.savefig(OUTPUT_DIR / "classification_roc_all_models.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(nn_classifier.loss_curve_)
ax.set_title("MLPClassifier training loss curve")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
fig.savefig(OUTPUT_DIR / "mlp_loss_curve.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.close(fig)

nn_regressor=MLPRegressor(hidden_layer_sizes=(64,32),activation="relu",max_iter=500,early_stopping=True,random_state=RANDOM_STATE)
reg_scaler=StandardScaler().fit(Xr_train)
y_scaler=StandardScaler().fit(yr_train.to_numpy().reshape(-1,1))
nn_regressor.fit(reg_scaler.transform(Xr_train),y_scaler.transform(yr_train.to_numpy().reshape(-1,1)).ravel())
nn_reg_pred=y_scaler.inverse_transform(nn_regressor.predict(reg_scaler.transform(Xr_test)).reshape(-1,1)).ravel()
add_regression_result("MLPRegressor (64, 32)",yr_test,nn_reg_pred)
reg_models["MLPRegressor (64, 32)"]=(nn_regressor,(reg_scaler,y_scaler,reg_features))
reg_table=pd.DataFrame(reg_results).sort_values(["RMSE","MAE"],ascending=True).reset_index(drop=True)
class_table=pd.DataFrame(class_results).sort_values(["Recall","F1"],ascending=False).reset_index(drop=True)
best_reg_name=reg_table.iloc[0]["Model"]
best_class_name=class_table.iloc[0]["Model"]
print("Regression comparison including neural network:\n")
print(reg_table.to_string(index=False, float_format="{:.4f}".format))
print("Classification comparison including neural network:\n")
print(class_table.to_string(index=False, float_format="{:.4f}".format))
print("Best alert classification report:\n",classification_report(yc_test,class_predictions[best_class_name],zero_division=0))

# %% [markdown]
# Section 6 selects the lowest-RMSE regression model and the highest-recall
# alert model (F1 breaks ties). AQI depends on pollutants, so some high scores
# are expected. This daily city-level dataset limits generalization.
# %%
print("\nSECTION 6: Comparison and Conclusion")
print("Best regression model:",best_reg_name)
print("Best alert model:",best_class_name)
print("Limitations: AQI is derived from pollutant readings; data are daily and city-level.")
print("Future scope: time-series forecasting can be studied later as a separate extension.")
print("Future scope: live sensor/API integration can be studied later as a separate extension.")
print("Future scope: mobile alerts can be studied later as a separate extension.")

# %% [markdown]
# Section 7 provides a callable prediction demo. It uses the selected trained
# AQI model and the classification decision threshold for the alert message.
# %%
category_bins=[0,50,100,200,300,400,np.inf]
category_labels=["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]

def predict_air_quality(pm25, pm10, no, no2, nh3, co, so2, o3, city, month):
    """Return predicted AQI, predicted category, and alert message."""
    city_text=str(city)
    city_code=int(city_encoder.transform([city_text])[0]) if city_text in city_encoder.classes_ else int(df["City_Encoded"].mode()[0])
    values={"PM2.5":pm25,"PM10":pm10,"NO":no,"NO2":no2,"NH3":nh3,"CO":co,"SO2":so2,"O3":o3,
            "Year":int(df["Year"].median()),"Month":int(month),"City_Encoded":city_code}
    for col in pollutants:
        if col not in values:
            values[col]=float(df[col].median())
    row=pd.DataFrame([[values[c] for c in reg_features]],columns=reg_features)
    if best_reg_name=="MLPRegressor (64, 32)":
        model,(scaler,target_scaler,features)=reg_models[best_reg_name]
        predicted=float(target_scaler.inverse_transform(model.predict(scaler.transform(row[features])).reshape(-1,1))[0,0])
    elif best_reg_name.startswith("Polynomial"):
        model,(poly,features)=reg_models[best_reg_name]
        predicted=float(model.predict(poly.transform(row[features]))[0])
    else:
        model,features=reg_models[best_reg_name]
        predicted=float(model.predict(row[features])[0])
    predicted=max(0.0,predicted)
    category=pd.cut(pd.Series([predicted]),bins=category_bins,labels=category_labels,right=True,include_lowest=True).iloc[0]
    message="ALERT: Air quality is POOR, avoid outdoor activity" if predicted>200 else "Air quality is acceptable"
    return {"predicted_AQI":round(predicted,2),"predicted_category":str(category),"message":message}

examples={
    "Clean-air example":(8,18,4,8,5,0.2,4,20,"Delhi",2),
    "Moderate example":(65,110,25,35,18,0.8,20,45,"Delhi",6),
    "Very-polluted example":(250,380,90,100,60,3.2,80,100,"Delhi",11)
}
for label,args in examples.items(): print(label+":",predict_air_quality(*args))

# %% [markdown]
# Section 8 explains why prediction remains useful even though AQI has a
# formula: stations can have missing or faulty sensors, and models allow
# comparison of estimates from partial readings.
# %%
print("\nViva note: Real stations may have missing or faulty sensors. A model can")
print("approximate AQI from partial readings and lets us compare prediction methods.")

# %% [markdown]
# This final cell saves the executed comparison tables, alert percentages, and
# summary values so the report can use real results from this run.
# %%
reg_table.to_csv(OUTPUT_DIR/"regression_comparison.csv",index=False)
class_table.to_csv(OUTPUT_DIR/"classification_comparison.csv",index=False)
k_table.to_csv(OUTPUT_DIR/"knn_k_accuracy.csv",index=False)
best_reg_row=reg_table.iloc[0].to_dict()
best_class_row=class_table.iloc[0].to_dict()
results={"project_title":"AirGuard India: Air Quality Analysis and Pollution Alert Prediction",
 "student":{"name":STUDENT_NAME,"roll_no":ROLL_NO,"section":SECTION,"teacher":TEACHER,"university":UNIVERSITY,"course":"INT234 Predictive Analytics (Python)"},
 "dataset":{"file":"data/city_day.csv","raw_rows":int(raw.shape[0]),"raw_columns":list(raw.columns),"clean_rows":int(len(df)),"clean_columns":int(df.shape[1]),"dropped_high_missing_columns":drop_cols,"alert_percent":float(df["ALERT"].mean()*100)},
 "best_regression":{"model":str(best_reg_name),**{k:float(v) for k,v in best_reg_row.items() if k!="Model"}},
 "best_classification":{"model":str(best_class_name),**{k:float(v) for k,v in best_class_row.items() if k!="Model"}},
 "top_correlations":{str(k):float(v) for k,v in pollutant_corr.reindex(pollutant_corr.abs().sort_values(ascending=False).index).head(5).items()},
 "regression_models":reg_table.to_dict(orient="records"),"classification_models":class_table.to_dict(orient="records"),
 "alert_percentages_by_city":{str(k):float(v) for k,v in alert_pct_city.items()},
 "prediction_examples":{k:predict_air_quality(*v) for k,v in examples.items()},"best_knn_k":int(best_k)}
with open(OUTPUT_DIR/"results.json","w",encoding="utf-8") as f: json.dump(results,f,indent=2,allow_nan=False)
print("Saved results.json and CSV comparison tables in",OUTPUT_DIR)
print("PNG charts created:",len(list(OUTPUT_DIR.glob("*.png"))))

