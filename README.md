## Chronic Kidney Disease Prediction

### Project Overview
This project aims to predict the occurrence of chronic kidney disease (CKD) using a dataset with various medical attributes. The objective is to build a reliable model to classify whether a patient has CKD or not.

### Technical Goal
From a technical perspective, the project involves:

* Developing a robust machine learning model to predict CKD based on medical attributes
* Handling missing data and performing feature engineering
* Evaluating different classification algorithms to achieve optimal performance
* Minimizing the classification error rate to enhance prediction accuracy

### Dataset 
The dataset used for this project is sourced from the UCI Machine Learning Repository and contains medical records for chronic kidney disease. It contains 400 entries, 24 features.
Dataset URL: [Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)

### Approach
The approach to solving this challenge includes:

* Exploratory Data Analysis (EDA): To understand data distribution, correlation patterns, and potential insights.
* Preprocessing and Feature Engineering: Data cleaning, handling missing values, label encoding, and datetime handling where applicable.
* Modeling: Training multiple models to identify the best performer.
* Feature Importance Analysis: Determining which features most significantly influence CKD prediction.

| Index | Model                       | Accuracy | Confusion Matrix        |
|-------|-----------------------------|----------|-------------------------|
| 1     | RandomForestClassifier      | 1.00000  | [[76, 0], [0, 44]]      |
| 6     | ExtraTreesClassifier        | 1.00000  | [[76, 0], [0, 44]]      |
| 2     | AdaBoostClassifier          | 0.99167  | [[75, 1], [0, 44]]      |
| 3     | GradientBoostingClassifier  | 0.98333  | [[74, 2], [0, 44]]      |
| 4     | XGBClassifier               | 0.98333  | [[74, 2], [0, 44]]      |
| 5     | CatBoostClassifier          | 0.98333  | [[74, 2], [0, 44]]      |
| 7     | LGBMClassifier              | 0.98333  | [[74, 2], [0, 44]]      |
| 0     | DecisionTreeClassifier      | 0.96667  | [[73, 3], [1, 43]]      |
| 8     | KNeighborsClassifier        | 0.95833  | [[72, 4], [1, 43]]      |

#### Packages Used
* Data Processing: pandas, numpy
* Visualization: matplotlib, seaborn, plotly
* EDA Summarization: skimpy
* Modeling: scikit-learn, xgboost, catboost, lightgbm
