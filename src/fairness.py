import os
import sys
import pickle 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# project path so other project files can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load saved models from the results folder

def load_saved_models(folder="results"):
    models ={} #empty dictionary but it will store all the loaded models
    for filename in os.listdir(folder):
        if filename.endswith(".pkl"):# Only load files with the .pkl extension
            filepath = os.path.join(folder, filename)
            name = filename.replace(".pkl","").replace("_"," ")
            with open(filepath,"rb")as f:#rb=read in binary mode
                models[name] = pickle.load(f)
    return models # return all loaded models

def calculate_fairness_metrics(y_true, y_pred, protected_attr,group_name):
    results = []#empty list to store the fairness results for each group
    groups = sorted(protected_attr.unique())
#repeat calculations for the same protected group separately
    for group in groups:
#checks each value in protected_attr If value= the current group it returns True otherwise False
        mask = protected_attr == group#
        cm = confusion_matrix(y_true[mask], y_pred[mask])

         
        tn, fp, fn,tp= cm.ravel() #extracting the value of confusion matrix 
#use confusion matrix values to calculate performance metrics
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0 #False Positive Rate(fpr)
        fnr= fn/(fn+tp) if (fn+tp)>0 else 0 #False Negative Rate (fnr)
        precision=tp/(tp+fp) if (tp+fp)> 0 else 0
        recall = tp/(tp+fn) if (tp+fn)> 0 else 0
        #store the calculated metrics for the current group
        results.append({
            "Group": group,
            "Count": int(mask.sum()),
            "FPR": round(fpr, 4),
            "FNR": round(fnr, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4)
        })
    df_results = pd.DataFrame(results)#converts results list of dictionaries into DataFrame
    disparities = {
        "FPR_Disparity": round(df_results["FPR"].max() - df_results["FPR"].min(), 4), 
        "FNR_Disparity": round(df_results["FNR"].max() - df_results["FNR"].min(), 4), 
        "Precision_Disparity": round(df_results["Precision"].max() - df_results["Precision"].min(), 4), 
        "Recall_Disparity": round(df_results["Recall"].max() - df_results["Recall"].min(), 4), 
    }
    #return the group performance results and the fairness disparity so they can be used later
    return df_results, disparities

def plot_fairness_comparison(all_results, model_name, save_dir="results"):
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    fig.suptitle(f"{model_name} - Fairness Metrics by Group", fontsize=16, fontweight="bold")
    metrics = ["FPR","FNR", "Precision", "Recall"]
    titles=[
        "False Positive Rate(Incorrect flagged as high risk)",
        "False Negative Rate(Guilty missed by model)",
        "Precision(% of flagged who actually recidivate)",
        "Recall (% of actual recidivists correctly flagged)"
    ]
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx// 2][idx % 2]
        for df_res in all_results:
            ax.bar(df_res["Group"], df_res[metric], alpha=0.7, label=df_res["Model"])
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(metric)
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_path = os.path.join(save_dir,f"{model_name.replace(' ','_')}_fairness.png")
    plt.savefig(save_path,dpi=150, bbox_inches="tight")
    plt.close('all')
    print(f"Saved fairness plot: {save_path}")
if __name__ == "__main__":
    from src.data_loader import load_compas_data
    from src.preprocessing import preprocess_compas

     # Load data
    print("Loading data..")
    df = load_compas_data()
    x_train, x_test,y_train, y_test, prot_train, prot_test = preprocess_compas(df)

    print("Loading models..")
    models = load_saved_models()
    
    y_test = pd.Series(y_test.values, index=prot_test.index)

    #evaluation for fairness for each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Fairness evaluation: {model_name}")
        print(f"{'='*60}")
        y_pred = model.predict(x_test)
        y_pred = pd.Series(y_pred, index=prot_test.index)
        all_results=[]

        #check fairness across diffrent race groups
        print("\n--- Race ---" )
        race_results, race_disp = calculate_fairness_metrics(
        y_test, y_pred, prot_test["race"], "Race"
        )
        #print fairness results for each group(race)
        print(race_results.to_string(index=False))
        print(f"\nDisparities: {race_disp}")
        race_results["Model"]= model_name
        all_results.append(race_results)


        #check fairness across diffrent sex groups
        print("\n--- Sex ---" )
        sex_results, sex_disp = calculate_fairness_metrics(
        y_test, y_pred, prot_test["sex"], "Sex"
        )
       #print fairness results for each group(sex)
        print(sex_results.to_string(index=False))
        print(f"\nDisparities: {sex_disp}")
        sex_results["Model"]= model_name
        all_results.append(sex_results)
        #visualization of the model's fairness performance
        plot_fairness_comparison(all_results, model_name)
