import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_compas(df, test_size=0.2,random_state=42):
        features=["age", "juv_fel_count", "juv_misd_count",
                   "juv_other_count", "priors_count", "decile_score"]
        target= "two_year_recid"
        protected=["sex","race"]

        df_clean=df[features +[target]+protected ].dropna().copy()
        protected_all=df_clean[protected].reset_index(drop=True)

        df_clean["sex"] = df_clean["sex"].map({"Female": 0, "Male": 1})
        df_clean = pd.get_dummies(df_clean, columns=["race"], prefix="race")
        x=df_clean.drop(columns=[target])
        y=df_clean[target]

        x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
)
        prot_train=protected_all.loc[x_train.index].reset_index(drop=True)
        prot_test=protected_all.loc[x_test.index].reset_index(drop=True)

        print(f"Train: {len(x_train)} | Test: {len(x_test)}")
        return x_train,x_test,y_train,y_test,prot_train,prot_test
if __name__=="__main__":
     from src.data_loader import load_compas_data
     df=load_compas_data()
     preprocess_compas(df)