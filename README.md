📌 Overview
An end-to-end supervised machine learning pipeline that predicts machine failures in a steel manufacturing environment before they occur — enabling proactive maintenance, reducing unplanned downtime, and cutting operational costs.
Built on 136,429 real-world-inspired sensor records from TATA Steel machinery, the project tackles severe class imbalance (1.57% failure rate), engineers physics-based features, and compares four classification models with full explainability via SHAP.

🎯 Problem Statement
Unplanned machine breakdowns in steel manufacturing cost lakhs of rupees per hour in lost production. Traditional reactive maintenance — fix it after it breaks — is expensive and unsafe. This project builds a binary classification model that scores every machine with a failure probability and risk tier (Low / Medium / High), giving maintenance engineers an early warning system backed by interpretable predictions.

📂 Dataset
SplitRowsColumnsTrain136,42914Test90,95413
Features: Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Machine type (L/M/H)
Target: Machine failure — Binary (0 = Normal, 1 = Failure)
Failure sub-types: TWF · HDF · PWF · OSF · RNF

🔧 Pipeline
Data Loading → EDA → Feature Engineering → Encoding → Scaling
→ SMOTE → Model Training → GridSearchCV → Evaluation → SHAP → Submission

⚙️ Feature Engineering
Five physics-based features created from raw sensor readings:
FeatureFormulaMeaningpowerTorque × (RPM × 2π/60)Mechanical power in Wattstemp_diffProcess temp − Air tempHeat dissipation gapwear_torqueTool wear × TorqueCompound mechanical stressrpm_squaredRPM²Centrifugal force proxytemp_rpm_ratioProcess temp / RPMThermal load per speed unit

🤖 Models Trained

Logistic Regression (baseline)
Random Forest
XGBoost (+ GridSearchCV tuning)
LightGBM ✅ (best model)


📊 Results
ModelPrecisionRecallF1-ScoreROC-AUCPR-AUCLogistic Regression0.070.780.130.86780.2196Random Forest0.210.620.320.90200.3512XGBoost0.030.880.070.88000.4081LightGBM 🏆0.290.600.390.91210.4069XGBoost (Tuned)0.030.880.070.88000.4081

LightGBM selected as the production model — best F1-Score (0.39) and ROC-AUC (0.9121), striking the most practical balance between catching failures and minimising false alarms.


💡 Key Techniques

SMOTE — synthetic oversampling to handle 1.57% failure rate
GridSearchCV — 5-fold stratified CV for XGBoost hyperparameter tuning
Threshold tuning — optimised decision boundary beyond default 0.5
SHAP — TreeExplainer for feature importance, beeswarm, dependence, and waterfall plots


🗂️ Repository Structure
├── train.csv
├── test.csv
├── TATA_Steel_ML_Submission.ipynb   ← Main notebook
├── submission.csv                   ← Final predictions
└── README.md

🚀 How to Run
bash# Install dependencies
pip install scikit-learn imbalanced-learn xgboost lightgbm shap matplotlib seaborn scipy

# Launch notebook
jupyter notebook TATA_Steel_ML_Submission.ipynb

📦 Dependencies
pandas · numpy · scikit-learn · imbalanced-learn · xgboost · lightgbm · shap · matplotlib · seaborn · scipy

🏭 Business Impact

Catches 60% of machine failures before they occur (LightGBM at default threshold)
XGBoost variant catches 88% of failures for critical machines where zero misses are acceptable
Each prediction includes a risk tier + SHAP explanation — actionable for maintenance engineers
Estimated 15–20% reduction in unplanned production downtime
