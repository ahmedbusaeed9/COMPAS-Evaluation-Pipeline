
import os 
import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import(
    confusion_matrix,roc_curve,auc,precision_recall_curve,
    classification_report, ConfusionMatrixDisplay
)
# project path so other project files can be imported
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load trained models from the results folder
def load_saved_models(folder="results"):
    models={}# Empty dictionary to store the loaded models
    if not os.path.exists(folder):# Check if the folder exists
        # if the folder does not exist stop the program
        raise FileNotFoundError(f"Results folder'{folder}' not found")
#loop to check each file one by one in the folder
    for filename in os.listdir(folder):
        #Only files ending with .pkl will be loaded others will be ignored
        if filename.endswith(".pkl"):
            filepath=os.path.join(folder,filename)
            #Cleans the filename to make it readable
            name=filename.replace(".pkl","").replace("_"," ")
            #Load saved models using pickle and store them in the models dictionary
            with open(filepath,"rb")as f:
                models[name]= pickle.load(f)
                print(f"loaded: {name}")
# Return the loaded models so they can be used later
    return models

#  trained model to predict test data and get probabilities for class 1
def evaluate_model(model,x_test,y_test,model_name):
    y_pred=model.predict(x_test)
    y_prob=model.predict_proba(x_test)[:, 1]
    #the evaluation results of the model and classification report
    print(f"\n{'='*60}")
    print(f"Evaluation: {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_test,y_pred,target_names=[" No Recidivism", "Recidivism"]))
    #confuision matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm= confusion_matrix(y_test,y_pred)
    ConfusionMatrixDisplay(cm, display_labels=[" No Recidivism", "Recidivism"]).plot(ax=axes[0],cmap="Blues")
    axes[0].set_title(f"{model_name} - Confusion Matrix")

   #ROC curve
    fpr, tpr, _ = roc_curve(y_test,y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"{model_name} - ROC Curve")
    axes[1].legend(loc="lower right")
    
    #Precision Recall Curve to evaluate how well the model identifies the positive class(Recidivism)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    axes[2].plot(recall, precision, color="green", lw=2, label=f"AUC = {pr_auc:.4f}")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title(f"{model_name} - Precision-Recall Curve")
    axes[2].legend(loc="lower left")
    plt.tight_layout()
    save_path = os.path.join("results", f"{model_name.replace(' ','_')}_evaluation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close('all') 
    print(f"saved plot: {save_path}")
#checks if the file is being run directly
if __name__ == "__main__":
    from src.data_loader import load_compas_data
    from src.preprocessing import preprocess_compas
    df = load_compas_data()
    x_train, x_test, y_train, y_test, prot_train, prot_test =preprocess_compas(df)
     # Load trained models from the results folder
    models = load_saved_models()
    #goes through each saved model one by one
    for name, model in models.items():
        #sends each model to the evaluation function.
        evaluate_model(model, x_test, y_test, name)
