import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from Data.load_data import load_tickets
from src.features import prepare_text, build_preprocessor

MODEL_PATH = "models/logreg_model.pkl"

def train():
    df = load_tickets()
    df = prepare_text(df)
    
    y=df["priority"]

    tag_cols=[c for c in df.columns if c.startswith("tag") ]
    
    drop_cols = ["priority", "body", "subject", "answer", "text"]

    X=df.drop(columns=drop_cols + tag_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42, test_size = 0.2)

    logreg_model = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=400, class_weight="balanced"))
    ])

    model_rf = Pipeline([("preprocess", build_preprocessor()),
                         ("clf", RandomForestClassifier(n_estimators = 300, random_state =42, class_weight="balanced_subsample", n_jobs=-1 ))])

    logreg_model.fit(X_train, y_train)

    joblib.dump(logreg_model, MODEL_PATH)

    joblib.dump(model_rf.fit(X_train, y_train), "models/model_rf.pkl")



    print(f"Model trained and saved to {MODEL_PATH}")
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")


if __name__ == "__main__":
    train()
  



