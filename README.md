# Credit Risk Prediction (ML Pipeline)

## 📌 Project Overview
This project builds an **end-to-end machine learning pipeline** to predict the likelihood of loan default.  
The goal is to show a complete data science workflow from raw data to deployable model with a focus on **imbalanced learning** and **model evaluation**.

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
- Logistic Regression → PR-AUC ≈ XX, F1 ≈ XX  
- Random Forest → PR-AUC ≈ XX, F1 ≈ XX  
- XGBoost → **Best model** with PR-AUC ≈ XX and F1 ≈ XX  

---

## Tech Stack
- Python (pandas, numpy, matplotlib, seaborn)  
- scikit-learn  
- imbalanced-learn  
- XGBoost  
- joblib  

