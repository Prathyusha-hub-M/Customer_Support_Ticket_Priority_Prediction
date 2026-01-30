import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

from Data.load_data import load_tickets
from src.features import prepare_text

MODEL_PATH = "models/logreg_model.pkl"

def evaluate():
    df = load_tickets()
    df = prepare_text(df)

    y=df["priority"]

    tag_cols=[c for c in df.columns if c.startswith("tag") ]
    
    drop_cols = ["priority", "body", "subject", "answer", "text"]

    X=df.drop(columns=drop_cols + tag_cols)

    logreg_model = joblib.load(MODEL_PATH)
    model_rf = joblib.load("models/model_rf.pkl")

    logreg_preds = logreg_model.predict(X)
    rf_preds = model_rf.predict(X)


    print("\n=== Logistic Regression (Balanced) ===")
    print(classification_report(y, logreg_preds))

    print("\n=== Random Forest (Balanced) ===")
    print(classification_report(y, rf_preds))

    # Confusion matrix for final model (Random Forest)
    labels = model_rf.named_steps["clf"].classes_
    cm = confusion_matrix(y, rf_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns = labels)

    print("\n=== Confusion Matrix: Random Forest ===")
    print(cm_df)



if __name__ == "__main__":
    evaluate()
