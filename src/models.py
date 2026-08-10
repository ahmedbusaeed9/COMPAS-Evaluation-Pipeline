import os 
import sys
import pickle 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
#the project root directory path so we can import modules from
# other project files
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def train_models(x_train,y_train,x_test,y_test):
# Create a dictionary containing three classification models
# so we can train, test, and compare them easily
    models={
        "LogisticRegression":LogisticRegression(max_iter=1000,random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42),
        "Gradient Boosting":GradientBoostingClassifier(n_estimators=100,random_state=42)
    }
# Create empty dictionaries to store trained models, predictions, and evaluation results
    trained_models={}
    predictions={}
    metrics={}#evaluates the score of the model
    print("Training models...")
#"Takes each model in dictionary,trains it,
# stores the trained model in trained_models
    for name,model in models.items():
        print(f"\n training{name}...")
        model.fit(x_train,y_train)
        trained_models[name]=model
#trained model predicts test data and saves both the final predicted classes and
#the probabilities to evaluate the model.
        y_pred=model.predict(x_test)
        y_prob=model.predict_proba(x_test)[:,1]#it gets the probability of class 1(positive class)
        #Stores both results in the predictions dictionary.
        predictions[name]={"pred":y_pred,"prob":y_prob}
#evaluates the trained model using accuracy, ROC-AUC,
#then stores the scores in the metrics dictionary
        acc=accuracy_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_prob)
        metrics[name]={"accuracy":acc,"auc_roc":auc}
        print(f"{name}|accuracy: {acc:.4f}|AUC-ROC:{auc:.4f}")

#return the three dictionaries to the place where the function was called
    return trained_models,predictions,metrics
#save the trained models to computer so they can be used later without training again
def save_models(trained_models,folder="results"):

     os.makedirs(folder, exist_ok=True)
     for name, model in trained_models.items():
        filepath = os.path.join(folder, f"{name.replace(' ','_')}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"saved: {filepath}")

#run the code bellow if file executed directly 
if __name__ == "__main__":
    from src.data_loader import load_compas_data
    from src.preprocessing import preprocess_compas
# Load and preprocess data
    df= load_compas_data()
#Prepares the COMPAS data,splits it into training/testing  data and protected attributes
    x_train, x_test, y_train, y_test, prot_train, prot_test = preprocess_compas(df)

# Train models
    trained_models,predictions,metrics=train_models(x_train,y_train,x_test,y_test)
#Call the function to save the trained ML models to files         
    save_models(trained_models)
    print("\n"+"="*50)
    print("Model Comparison Summary")
    print("-"*50)
    for name, m in metrics.items():
        print(f"{name:25s}|Acc: {m['accuracy']:.4f}|AUC: {m['auc_roc']:.4f}")