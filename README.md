# Credit Risk Prediction (ML Pipeline)

##  Project Overview
I chose to work on credit risk prediction because it’s a real-world problem that affects both banks and individuals, predicting whether someone will default on a loan is a key challenge in finance. In this project, I built an end-to-end machine learning pipeline that goes from raw data all the way to a deployable model. Along the way, I focused on handling imbalanced data, applying different models, and carefully evaluating performance to make the predictions more reliable.
---

##  Steps in the Project
1. **ETL & Data Cleaning**  
   - Removed duplicates, handled missing values  
   - Prepared numerical and categorical features  

2. **Preprocessing & Sampling**  
   - Median/Mode imputation  
   - Scaling & One-Hot Encoding  
   - Handled class imbalance with **RandomOverSampler** inside pipelines  

3. **Modeling**  
   - Compared three classifiers: **Logistic Regression, Random Forest, and XGBoost**  

4. **Hyperparameter Tuning**  
   - Used **RandomizedSearchCV** with **PR-AUC scoring**  
   - Ensured robust cross-validation  

5. **Evaluation**  
   - Precision-Recall & ROC Curves  
   - Confusion Matrices  
   - F1-based threshold optimization (instead of fixed 0.5 cutoff)  

6. **Deployment**  
   - Exported final pipeline with preprocessing + best model + tuned threshold  

---

## Results
- XGBoost → **Best model** with PR-AUC = 0.8935 and F1 = 0.8145 

---

## Tech Stack
- Python (pandas, numpy, matplotlib, seaborn)  
- scikit-learn  
- imbalanced-learn  
- XGBoost  
- joblib  

