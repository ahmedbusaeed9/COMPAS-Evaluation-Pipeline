import os 
import sys 
import pickle
import shap
import lime 
import lime.lime_tabular
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_saved_model(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def generate_shap_explanation(model,x_test, feature_names, save_dir="results"):
    print("generating SHAP explanation..")
     # Use TreeExplainer for tree-based models (faster than KernelExplainer)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, x_test, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Global)", fontsize=16, fontweight="bold")

    save_path = os.path.join(save_dir, "shap_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"saved SHAP plot: {save_path}")

def generate_lime_explanation(model, x_test, y_test, feature_names, class_names, instance_idx=0, save_dir="results"):
    print(f"Generating LIME explanation for instance {instance_idx}")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        x_test.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=42
    )
    # explain one prediction made by the model
    exp = explainer.explain_instance(
        x_test.values[instance_idx],
        model.predict_proba,
        num_features=10
    )
    # save explanation as HTML file(Github purpose)
    save_html = os.path.join(save_dir, f"lime_explanation_instance_{instance_idx}.html")
    exp.save_to_file(save_html)
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Explanation - instance{instance_idx}",fontsize=14,fontweight="bold")
    save_png = os.path.join(save_dir, f"lime_explanation_instance_{instance_idx}.png")
    plt.savefig(save_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" saved LIME html: {save_html}")
    print(f" saved LIME png: {save_png}")

    true_label = y_test.iloc[instance_idx] if hasattr(y_test, 'iloc') else y_test[instance_idx]
    pred_prob = model.predict_proba(x_test.values[instance_idx:instance_idx+1])[0]
    print(f"\n Instance {instance_idx}:")
    print(f" True Label: {'Recidivism' if true_label ==1 else 'No Recidivism'}")
    print(f" Predicted Prob: No Recidivism={pred_prob[0]:.3f}, Recidivism={pred_prob[1]:.3f}")

if __name__=="__main__":
    from src.data_loader import load_compas_data
    from src.preprocessing import preprocess_compas
    
    print("Loading data...")
    df = load_compas_data() #loads the COMPAS dataset into DataFrame
    #preprocesses the dataset and splits it into training and testing sets
    x_train, x_test, y_train, y_test, prot_train, prot_test = preprocess_compas(df)
    # load the trained Gradient Boosting model
    model_path = os.path.join("results", "Gradient_Boosting.pkl")
    print(f" Loading model: {model_path}")
    model = load_saved_model(model_path)
    # Get the names of the input features from the test dataset
    feature_names = list(x_test.columns)
    class_names =["No Recidivism", "Recidivism"]

    # Generate explanations
    print("\n" + "="*60)
    print(" Model Explainability Analysis")
    print("="*60)
    #generate global SHAP explanation (which features are important overall for the model)
    generate_shap_explanation(model, x_test, feature_names)

    # LIME shows why the model made a prediction for (one specific person)
    generate_lime_explanation(model, x_test, y_test, feature_names, class_names,instance_idx=0)

    print("\n The explainability analysis complete")
