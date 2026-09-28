# ❤️ Heart Attack Risk Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-oversampling-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)
![Status](https://img.shields.io/badge/Purpose-Educational-lightgrey.svg)

An end-to-end machine learning project that predicts heart attack risk from patient health, lifestyle, and demographic data. The notebook covers data cleaning, exploratory analysis, class balancing, several feature-selection strategies, a wide comparison of classifiers and ensembles, and a final leakage-aware validation with a tuned Random Forest.

> ⚠️ **Disclaimer:** This is an educational / research project. It is **not** a medical diagnostic tool and must not be used to make clinical decisions.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Feature Selection Methods](#-feature-selection-methods)
- [Models Evaluated](#-models-evaluated)
- [Results](#-results)
- [Enhanced Final Validation](#-enhanced-final-validation)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Limitations and Honest Notes](#-limitations-and-honest-notes)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🔎 Overview

The goal is to classify whether a patient is at risk of a heart attack (`Heart Attack Risk`: `1` = risk, `0` = no risk) and to compare how different preprocessing, feature-selection, and modelling approaches affect performance.

**Key highlights**

- Full cleaning pipeline (duplicate removal, blood-pressure splitting, encoding of categorical features)
- Exploratory analysis with pie charts, category plots, count plots, and correlation heatmaps
- Class-imbalance handling using **Random Over Sampling**
- Six feature-selection / model-combination strategies: Chi-Squared, Mutual Information, Lasso, PCA, RFE, plus ensemble methods (Stacking, Max Voting)
- Hyperparameter comparisons for KNN, Random Forest, Decision Tree, Gradient Boosting, Naive Bayes, Logistic Regression, and SVM
- An **enhanced validation section** that splits the data *before* oversampling, tunes a Random Forest with `GridSearchCV`, runs 5-fold cross-validation, and adds a confusion matrix, ROC curve, feature importance, and a reusable prediction function

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `heart_attack_prediction_dataset.csv` |
| Records | 8,763 patients |
| Original columns | 26 |
| Target | `Heart Attack Risk` (binary) |
| Class balance (original) | ~64% no risk / ~36% risk |

> The dataset file is **not included** in this repository notebook upload. Place `heart_attack_prediction_dataset.csv` in the project root before running (see [Usage](#-usage)). Download it here: [heart_attack_prediction_dataset.csv](https://1drv.ms/x/c/0722dab5925dcfbd/IQAAFekereJtRYUHYrbdQvdYAUqoSRKJ6jRfUtif4KpSPsw?e=A7T4bD).

**Original features**

| Category | Features |
|---|---|
| Demographics | Age, Sex, Income, Country, Continent, Hemisphere, Patient ID |
| Clinical | Cholesterol, Blood Pressure (systolic/diastolic), Heart Rate, Diabetes, BMI, Triglycerides, Previous Heart Problems, Medication Use |
| Lifestyle | Smoking, Obesity, Alcohol Consumption, Exercise Hours Per Week, Diet, Stress Level, Sedentary Hours Per Day, Physical Activity Days Per Week, Sleep Hours Per Day |
| Genetic | Family History |
| Target | Heart Attack Risk |

---

## 🔄 Project Workflow

```
Raw CSV
  │
  ├─ 1. Data Cleaning
  │     • Drop duplicates
  │     • Drop Patient ID, Continent, Hemisphere
  │     • Sex → is_male (one-hot, 0/1)
  │     • Blood Pressure "sys/dia" → systolic_pressure, diastolic_pressure
  │     • Diet → ordinal (Unhealthy=0, Average=1, Healthy=2)
  │     • Country → LabelEncoder
  │
  ├─ 2. Exploratory Data Analysis
  │     • Target distribution, category plots, count plots
  │     • Correlation heatmap → drop BMI and Previous Heart Problems (weak correlation)
  │
  ├─ 3. Class Balancing
  │     • RandomOverSampler + MinMaxScaler
  │
  ├─ 4. Feature Selection + Modelling
  │     • Chi2 / Mutual Info (SelectKBest) + Random Forest
  │     • Lasso  • PCA  • RFE
  │     • Stacking  • Max Voting
  │
  ├─ 5. Visualisation and comparison of results
  │
  └─ 6. Enhanced Final Validation
        • Split first → oversample training set only → scale on train only
        • Compare 7 models → GridSearchCV on Random Forest
        • 5-fold CV, confusion matrix, ROC curve, feature importance
        • Prediction function
```

### Cleaning summary

- **Duplicates:** checked and removed (`df.drop_duplicates`)
- **Missing values:** none found in any feature
- **Engineered features:** `is_male`, `systolic_pressure`, `diastolic_pressure`
- **Dropped:** `Patient ID`, `Continent`, `Hemisphere` (irrelevant / redundant); `BMI`, `Previous Heart Problems` (low correlation with target)

---

## 🧪 Feature Selection Methods

| Method | Type | How it is used |
|---|---|---|
| Chi-Squared (`chi2`) | Filter | `SelectKBest`, k iterated from 1 to 21, scored with Random Forest |
| Mutual Information | Filter | `SelectKBest`, k iterated from 1 to 21, scored with Random Forest |
| Lasso | Embedded | Keeps features with non-zero coefficients (`alpha=0.001`), then trains several classifiers |
| PCA | Dimensionality reduction | Reduces to 12 components, then trains several classifiers |
| RFE | Wrapper | Recursive elimination down to 15 features, then trains several classifiers |

---

## 🤖 Models Evaluated

- K-Nearest Neighbors (Euclidean, Manhattan, Minkowski variants)
- Logistic Regression (`liblinear`, `saga`, L1 penalty)
- Decision Tree (`gini`, `log_loss`)
- Random Forest (multiple `n_estimators`)
- Gradient Boosting
- Gaussian Naive Bayes
- Support Vector Machine (RBF kernel)
- **Stacking Classifier** (8 base models, Random Forest as meta-learner)
- **Voting Classifier** (Decision Tree, Gradient Boosting, Random Forest, Naive Bayes, Logistic Regression)

Each model is evaluated on **Accuracy, Precision, Recall, F1-score, and ROC-AUC**.

---

## 📈 Results

Best result per approach in the original experiments (20% test split, `random_state=42`):

| Approach | Classifier | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Stacking** | Random Forest (meta) | **0.8209** | 0.9715 | 0.6628 | 0.7880 | 0.8216 |
| Chi-Squared (k=21) | Random Forest | 0.8031 | 0.8754 | 0.7088 | 0.7834 | **0.8333** |
| Mutual Information (k=21) | Random Forest | 0.8031 | 0.8754 | 0.7088 | 0.7834 | **0.8333** |
| RFE (15 features) | Random Forest | 0.7964 | 0.8700 | 0.6991 | 0.7753 | 0.7969 |
| Lasso (13 features) | Random Forest | 0.7653 | 0.7998 | 0.7106 | 0.7526 | 0.8234 |
| PCA (12 components) | Random Forest | 0.7538 | 0.7637 | 0.7381 | 0.7507 | 0.7538 |
| Max Voting | Ensemble | 0.6471 | 0.6525 | 0.6363 | 0.6443 | 0.6472 |

**Observations**

- Tree-based ensembles (Random Forest, Stacking) clearly outperformed the other model families.
- Logistic Regression and Gaussian Naive Bayes stayed close to 50% accuracy, and KNN and SVM were only slightly above it (roughly 54–57%).
- With Chi-Squared and Mutual Information, the best score came at k=21, meaning **all** remaining features were kept.
- Stacking had the highest precision (97%) but lower recall (66%), so it misses a meaningful share of at-risk patients, which is a costly failure mode in a health setting.

---

## ✅ Enhanced Final Validation

The original workflow oversampled the full dataset *before* splitting into train and test sets. Because random oversampling duplicates rows, copies of the same record can land in both sets, which inflates test scores. The final section of the notebook addresses this:

1. Stratified 80/20 split of the cleaned data
2. `RandomOverSampler` applied to the **training set only**
3. `MinMaxScaler` fitted on training data only
4. Comparison of 7 models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, KNN, SVM
5. `GridSearchCV` (5-fold, ROC-AUC) on Random Forest over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
6. 5-fold cross-validation of the tuned model
7. Confusion matrix, classification report, ROC curve
8. Top-15 feature importance chart
9. A `predict_heart_attack_risk()` helper for single-record prediction

> 📝 The notebook was saved without outputs for this section, so run it to reproduce the final tuned-model metrics. Add them here once you have them:
>
> | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
> |---|---|---|---|---|---|
> | Tuned Random Forest | _run notebook_ | _run notebook_ | _run notebook_ | _run notebook_ | _run notebook_ |

**Example prediction**

```python
example_input = X_final.iloc[0].to_dict()
print(predict_heart_attack_risk(example_input))
# {'Prediction': 0 or 1, 'Predicted probability of class 1': 0.xxxx, 'Result': '...'}
```

---

## 🛠️ Installation

**Requirements:** Python 3.9+ (developed with 3.11)

```bash
# 1. Clone the repository
git clone https://github.com/Dharmeshkumar2327/heart-attack-risk-prediction.git
cd heart-attack-risk-prediction

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
```

---

## ▶️ Usage

1. Download the [dataset](https://1drv.ms/x/c/0722dab5925dcfbd/IQAAFekereJtRYUHYrbdQvdYAUqoSRKJ6jRfUtif4KpSPsw?e=A7T4bD) and save it as `heart_attack_prediction_dataset.csv` in the project root.
2. Launch Jupyter:

   ```bash
   jupyter notebook heart-attack-prediction-enhanced.ipynb
   ```

3. Run all cells from top to bottom (**Kernel → Restart & Run All**).

The notebook also writes the cleaned, oversampled data to `heart_attack_prediction_dataset_after_cleaning.csv`.

> ⏱️ The stacking model, the k-loop feature selection, and `GridSearchCV` are the slowest steps. Expect several minutes on a laptop.

---

## 📁 Project Structure

```
.
├── heart-attack-prediction-enhanced.ipynb          # Main notebook
├── heart_attack_prediction_dataset.csv             # Input dataset (add manually)
├── heart_attack_prediction_dataset_after_cleaning.csv  # Generated by the notebook
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations and Honest Notes

- **Not for clinical use.** Results depend on this dataset, its preprocessing, and the chosen evaluation method.
- **Possible data leakage in the original experiments.** Oversampling before the train/test split means the headline numbers (e.g. 82.09% for Stacking) are likely optimistic. The enhanced section uses the safer split-first approach, and its numbers should be treated as the more trustworthy estimate.
- **Weak signal in several models.** Near-chance performance from Logistic Regression and Naive Bayes, and very low feature correlations with the target (the heatmap is scaled to ±0.02 to make them visible), suggest the features carry limited linear signal. Check how the dataset was generated before drawing real-world conclusions.
- **Single split.** The original comparisons use one train/test split rather than repeated cross-validation, so small differences between methods may not be meaningful.
- **Accuracy is not enough.** In a medical setting, recall (missing at-risk patients) matters. Review the confusion matrix and consider threshold tuning.
- **Feature importance is not causation.**
- **Some scores use hard predictions for ROC-AUC** (RFE, PCA, Stacking, Voting), which understates the true probability-based AUC.

---

## 🚀 Future Improvements

- Evaluate all original experiments with split-before-oversample (or use `imblearn.pipeline.Pipeline` so resampling happens inside each CV fold)
- Try SMOTE or class weights instead of random duplication
- Add XGBoost, LightGBM, and CatBoost
- Tune the decision threshold to prioritise recall
- Add SHAP explanations
- Validate on a real clinical dataset
- Wrap the model in a small Streamlit or Flask app

---

## 🤝 Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. Add a `LICENSE` file to the repository, or change this section to match your chosen license.

---

## 👤 Author

**Dharmesh Kumar**
- GitHub: [@Dharmeshkumar2327](https://github.com/Dharmeshkumar2327)

⭐ If you found this project useful, consider giving it a star!
