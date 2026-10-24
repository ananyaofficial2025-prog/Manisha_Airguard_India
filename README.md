# AirGuard India: Air Quality Analysis and Pollution Alert Prediction

A machine learning project that analyses air quality data of Indian cities, predicts the **AQI (Air Quality Index)**, and raises a **pollution alert** when the air quality is *Poor* or worse.

---

## Student Details

| Field | Details |
|---|---|
| **Submitted by** | Manisha |
| **Roll No.** | [Enter Roll No] |
| **Section** | [Enter Section] |
| **Course** | INT234: Predictive Analytics |
| **Teacher** | [Enter Teacher Name] |
| **University** | [Enter University] |

---

## Table of Contents

1. [Abstract](#abstract)
2. [Problem Statement](#problem-statement)
3. [Aim and Objectives](#aim-and-objectives)
4. [Dataset Description](#dataset-description)
5. [Tech Stack](#tech-stack)
6. [Syllabus Mapping](#syllabus-mapping)
7. [Methodology and Working](#methodology-and-working)
8. [Project Structure](#project-structure)
9. [How to Run](#how-to-run)
10. [Results](#results)
11. [Prediction Demo](#prediction-demo)
12. [Applications](#applications)
13. [Limitations](#limitations)
14. [Future Scope](#future-scope)
15. [Conclusion](#conclusion)
16. [References](#references)

---

## Abstract

Air pollution is one of the biggest health and environmental problems in India. This project, **AirGuard India**, uses the publicly available *Air Quality Data in India (2015-2020)* dataset to study how different pollutants (PM2.5, PM10, NO2, SO2, CO, O3 and others) affect air quality across Indian cities.

The project covers the complete predictive analytics workflow: data cleaning and preprocessing, exploratory data analysis (EDA), regression to predict the numerical AQI value, classification to predict whether a pollution alert should be raised, clustering to group days by pollution level, association rule mining to find pollution patterns, a foundational neural network, and model performance improvement using cross-validation and hyperparameter tuning.

---

## Problem Statement

People, schools, hospitals and government agencies need to know **when the air quality becomes harmful** so that they can take precautions. However, in real monitoring stations, sensors often fail or give missing readings.

This project builds a system that:

- analyses pollutant readings of Indian cities,
- predicts the AQI value from the available pollutant readings,
- classifies whether the day needs a **pollution alert** (Poor, Very Poor or Severe air quality).

---

## Aim and Objectives

### Aim
To build a predictive analytics system that analyses pollution data of Indian cities and predicts AQI and pollution alerts using machine learning techniques.

### Objectives

1. Clean and prepare real-world air quality data (handle missing values and outliers).
2. Perform EDA to understand which pollutants affect air quality the most.
3. Study pollution trends (yearly, monthly and city-wise comparison).
4. Predict the numerical AQI value using regression models.
5. Predict the pollution alert (Alert / No Alert) using classification models.
6. Group similar pollution days using clustering (unsupervised learning).
7. Discover pollution patterns using association rule mining.
8. Implement a foundational neural network for classification and regression.
9. Compare all models and improve their performance using cross-validation and hyperparameter tuning.
10. Build a simple prediction function that gives the AQI, category and alert message.

---

## Dataset Description

- **Name:** Air Quality Data in India (2015-2020)
- **Source:** Kaggle
- **File used:** `data/city_day.csv`
- **Type:** Daily air quality readings of Indian cities
- **Size:** about 29,000 rows

| Column | Description |
|---|---|
| `City` | Name of the Indian city |
| `Date` | Date of the reading |
| `PM2.5` | Fine particulate matter (diameter below 2.5 micrometres) |
| `PM10` | Coarse particulate matter (diameter below 10 micrometres) |
| `NO`, `NO2`, `NOx` | Nitrogen oxides |
| `NH3` | Ammonia |
| `CO` | Carbon monoxide |
| `SO2` | Sulphur dioxide |
| `O3` | Ozone |
| `Benzene`, `Toluene`, `Xylene` | Volatile organic compounds |
| `AQI` | Air Quality Index (numerical value) |
| `AQI_Bucket` | Air quality category (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) |

**Alert target created in this project:** `Alert = 1` if `AQI_Bucket` is Poor, Very Poor or Severe, otherwise `Alert = 0`.

---

## Tech Stack

| Category | Tool / Library | Purpose |
|---|---|---|
| Language | Python 3 | Main programming language |
| Environment | Jupyter Notebook | Writing and running the project |
| Data handling | pandas, NumPy | Loading, cleaning and processing data |
| Visualisation | Matplotlib, Seaborn | Charts, heatmaps and plots |
| Machine learning | scikit-learn | Regression, classification, clustering, neural network, tuning |
| Statistics | statsmodels | OLS regression summary |
| Hierarchical clustering | SciPy | Dendrogram |
| Association rules | mlxtend | Apriori algorithm |
| Report generation | python-docx | Project report (.docx) |
| Version control | Git and GitHub | Code hosting |
| AI coding assistant | Codex | Used to help generate the project code |

No API, no live data and no internet connection is needed to run the project. Only the local CSV file is used.

---

## Syllabus Mapping

| Course Outcome / Unit | Topic | Where it is used in this project |
|---|---|---|
| **Unit I** (CO1) | Predictive analytics, ML types, data preprocessing, EDA | Missing value handling, outlier capping, encoding, scaling, EDA charts, trend graphs |
| **Unit II** (CO2) | Regression and evaluation | Simple, Multiple and Polynomial Regression, OLS, MAE, MSE, RMSE, R2 |
| **Unit III** (CO3) | Classification and evaluation | Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM, Accuracy, Precision, Recall, F1, Log Loss, AUC, Confusion Matrix |
| **CO4** | Clustering and association rules | K-Means, Hierarchical clustering, Apriori |
| **CO5** | Foundational neural networks | MLPClassifier and MLPRegressor (feed-forward network) |
| **CO6** | Improving model performance | k-fold cross-validation, GridSearchCV, class balancing, overfitting vs underfitting |

---

## Methodology and Working

### Step 1: Data Preparation (Unit I)

1. Load `city_day.csv` and inspect it (`head`, `info`, `describe`, shape).
2. Calculate the missing value percentage of each column.
3. Drop columns with very high missing values (such as Xylene).
4. Drop rows where `AQI` is missing (the target is never imputed).
5. Fill the remaining missing pollutant values with the **city-wise median**.
6. Extract `Year` and `Month` from `Date`, and label-encode `City`.
7. Create the `Alert` target column.
8. Handle outliers using **IQR capping**.
9. Scale features using **StandardScaler** (fitted on training data only, to avoid data leakage).

### Step 2: Exploratory Data Analysis

- AQI distribution histogram
- Correlation heatmap (which pollutants affect AQI the most)
- City-wise average AQI comparison
- AQI category count plot
- PM2.5 vs AQI scatter plot
- Yearly and monthly AQI trends (seasonality) and top polluted cities
- Alert percentage by city

### Step 3: Regression (Unit II)

- **Target:** `AQI`
- Models: Simple Linear Regression (PM2.5 to AQI), Multiple Linear Regression, Polynomial Regression (degree 2 and 3)
- Metrics: MAE, MSE, RMSE, R2
- Plots: regression line, actual vs predicted, residual plot

### Step 4: Classification (Unit III)

- **Target:** `Alert` (binary)
- The `AQI` and `AQI_Bucket` columns are **removed from the features** to avoid target leakage.
- Models: Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVM
- Class balancing is used where supported.
- Metrics: Accuracy, Precision, Recall, F1, Log Loss, ROC-AUC, Confusion Matrix, ROC curves
- For an alert system, **Recall** on the alert class matters most, because missing a polluted day is worse than a false alarm.

### Step 5: Unsupervised Learning (CO4)

- **K-Means clustering:** elbow method and silhouette score to choose the number of clusters, followed by a cluster profile table.
- **Hierarchical clustering:** dendrogram and Agglomerative clustering on a sample of the data.
- **Apriori association rules:** pollutants are converted into Low / Medium / High levels, then rules (such as *High PM2.5 and Winter leads to Alert*) are found using support, confidence and lift.

### Step 6: Neural Network (CO5)

- `MLPClassifier` for alert prediction and `MLPRegressor` for AQI prediction.
- Architecture: two hidden layers (64 and 32 neurons), ReLU activation, early stopping.
- Loss curve is plotted to show learning.

### Step 7: Improving Model Performance (CO6)

- 5-fold cross-validation (stratified for classification)
- GridSearchCV for KNN, Decision Tree and SVM
- Before vs after tuning comparison
- Overfitting vs underfitting analysis using Decision Tree depth

### Step 8: Comparison and Prediction

- All models are compared in tables.
- The best regression and classification models are used in a prediction function.

### Workflow Diagram

```
Raw Data (city_day.csv)
        |
        v
Data Cleaning and Preprocessing
        |
        v
Exploratory Data Analysis and Trends
        |
        +--> Regression (predict AQI)
        +--> Classification (predict Alert)
        +--> Clustering and Association Rules
        +--> Neural Network
        |
        v
Model Tuning and Comparison
        |
        v
Prediction Demo (AQI + Category + Alert message)
```

---

## Project Structure

```
Manisha_AirGuard_India/
├── data/
│   └── city_day.csv                          # Dataset (Kaggle)
├── outputs/                                  # All saved charts (PNG)
├── Manisha_AirGuard_India.ipynb              # Main notebook (code + outputs)
├── Manisha_AirGuard_Project_Report.docx      # Project report
├── Viva_Questions_Answers.md                 # Viva preparation
├── README.md                                 # Project documentation
└── requirements.txt                          # Python libraries
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/NishantKumar-Dev/Nishant_AirGuard_India.git
cd Nishant_AirGuard_India
```

*(Use your actual repository link if the repository was renamed.)*

### 2. Install the requirements

```bash
pip install -r requirements.txt
```

### 3. Check the dataset

Make sure the file `data/city_day.csv` is present.

### 4. Open the notebook

```bash
jupyter notebook Manisha_AirGuard_India.ipynb
```

Then run all cells from top to bottom (**Kernel > Restart and Run All**).

Python 3.9 or above is recommended. The notebook runs in a few minutes on a normal laptop.

---

## Results

> Fill this table with the actual values from the executed notebook before submission.

### Regression

| Model | MAE | MSE | RMSE | R2 |
|---|---|---|---|---|
| Simple Linear Regression | | | | |
| Multiple Linear Regression | | | | |
| Polynomial Regression (degree 2) | | | | |
| Polynomial Regression (degree 3) | | | | |
| Neural Network (MLPRegressor) | | | | |

**Best regression model:** _fill after running_

### Classification (Alert Prediction)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| KNN | | | | | |
| Naive Bayes | | | | | |
| Decision Tree | | | | | |
| SVM | | | | | |
| Neural Network (MLPClassifier) | | | | | |

**Best classification model:** _fill after running_

### Clustering

| Method | Number of clusters | Silhouette score |
|---|---|---|
| K-Means | | |
| Hierarchical | | |

### Key Findings

- Most influential pollutants for AQI: _fill after running_
- Most polluted cities: _fill after running_
- Most polluted season: _fill after running_
- Top association rule: _fill after running_

---

## Prediction Demo

The notebook contains a function that takes pollutant readings and returns the result:

```python
predict_air_quality(pm25, pm10, no, no2, nh3, co, so2, o3, city, month)
```

**Output:**

- Predicted AQI value
- Predicted AQI category
- Pollution cluster group
- Alert message, for example: `ALERT: Air quality is POOR, avoid outdoor activity`

---

## Applications

- **Citizens:** know when to avoid outdoor activity
- **Schools:** decide on outdoor sports and assembly
- **Hospitals:** prepare for pollution-related health cases
- **Government and environment agencies:** issue alerts and plan policies
- **Monitoring stations:** estimate AQI when some sensors are missing or faulty

---

## Limitations

- AQI is calculated from pollutant values using a formula, so model accuracy is naturally high.
- The data is daily and city-level, not station-level or hourly.
- The dataset covers 2015-2020 only, so it does not reflect recent changes.
- The project does not forecast future AQI (time-series forecasting is outside the syllabus).

---

## Future Scope

- Time-series forecasting to predict AQI for upcoming days
- Integration with live sensor and API data
- Mobile app or SMS alerts for citizens
- Hourly and station-level analysis

---

## Why Machine Learning if AQI Has a Formula?

Real monitoring stations often have missing or faulty sensors. A machine learning model can approximate the AQI from partial readings, and it lets us compare the performance of different algorithms on a real-world problem.

---

## Conclusion

AirGuard India shows how predictive analytics can turn raw pollution data into useful information. The project covers data preprocessing, EDA, regression, classification, clustering, association rules, neural networks and model improvement, and finally gives an alert system that warns when air quality becomes harmful.

---

## References

1. Air Quality Data in India (2015-2020), Kaggle: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
2. Central Pollution Control Board (CPCB), National Air Quality Index: https://cpcb.nic.in
3. scikit-learn documentation: https://scikit-learn.org
4. pandas documentation: https://pandas.pydata.org
5. mlxtend documentation (Apriori): https://rasbt.github.io/mlxtend

---

## Acknowledgement

I thank my teacher **[Enter Teacher Name]** for the guidance and support during this project.

**Project by Manisha | INT234 Predictive Analytics**
