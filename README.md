# COMPAS Evaluation Pipeline

This project evaluates different machine learning models on the COMPAS dataset. It compares their performance, explains their predictions using SHAP and LIME, and checks whether the models behave fairly across different demographic groups, including race and sex.

## Project Structure
The project is divided into separate Python files, with each file handling one specific step of the machine learning (ML) process.
| File | Responsibility |
| :--- | :--- |
| `src/data_loader.py` | Loads the compas.csv dataset, checks the required columns, and shows the dataset size
 |
| `src/preprocessing.py` |Selects the required features, converts variables into a usable format, and splits the data into training and testing sets|
| `src/models.py` | Trains Logistic Regression, Random Forest, and Gradient Boosting models and saves them as .pkl files|
| `src/evaluation.py` |Creates confusion matrix, ROC curve, and PR curve plots to evaluate each model|
| `src/explainability.py` | Creates SHAP summaries to explain the models (overall) and LIME plots to explain (individual predictions)|
| `src/fairness.py` | Checks model performance across different race and sex groups using FPR, FNR, Precision, and Recall, and creates fairness charts |
| `results/` | Stores all the final outputs from the project, including trained models, evaluation plots, SHAP/LIME explanations, and fairness results|

## Requirements
-   **Python:** 3.11+
-   **Dependencies:**
    ```bash
    pip install pandas scikit-learn shap lime matplotlib
    ```
## Data Setup
Place the raw COMPAS dataset in the following location relative to the project root:
data/compas.csv

## How to Run

Run each step of the pipeline in order from the project root. Each step uses the output from the previous steps. Once the required files are available in `results/`, you can also run a step again whenever needed.

| Step | Command | Output |
| :--- | :--- | :--- |
| 1. Train Models | `python -m src.models` | Trained model files in `results/*.pkl` |
| 2. Evaluate Performance | `python -m src.evaluation` | Evaluation plots in `results/*_evaluation.png` |
| 3. Generate Explanations | `python -m src.explainability` | SHAP summary and LIME explanation files |
| 4. Evaluate Fairness | `python -m src.fairness` |Console tables + \results/*_fairness.png` charts for each model

## Key Findings Summary
**Performance**: Gradient Boosting achieves the best overall performance (Accuracy: 0.691, ROC-AUC: 0.744), slightly outperforming Logistic Regression. Random Forest performs worse, with lower accuracy (0.64) and higher false-positive rates across demographic groups.
**Fairness**: There is a clear difference in performance across racial groups. African-American defendants have about 2× higher False Positive Rates than Caucasian defendants. There is also a difference by sex, with men being more likely to be over-flagged and women more likely to be under-flagged.
**Explainability**: `priors_count` and `decile_score` are the most important features overall according to SHAP, for individual predictions (LIME), the analysis shows that `age` has a strong protective effect.

## Report 
See COMPAS Project Report.pdf for full methodology, results, and discussion.