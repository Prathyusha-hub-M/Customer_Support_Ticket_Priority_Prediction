from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

#Combining email subject and body into a single column to get more context and cleaning text columns
def prepare_text(df):
    df["subject"]= df["subject"].fillna("")
    df["body"]=df["body"].fillna("")
    df["text"]=df["subject"]+" "+df["body"]

    urgent_words = [
    "urgent", "asap", "immediately", "critical",
    "blocked", "locked", "cannot access", "can't access",
    "down", "outage", "payment failed", "failed",
    "error", "system down"
    ]

    complaint_words = [
    "problem", "issue", "bug", "error",
    "does not work", "doesn't work", "broken",
    "wrong", "complaint", "frustrated"
    ]

    inquiry_words = [
    "how", "can you", "could you",
    "question", "inquiry", "information",
    "help me", "please explain",
    "feature request", "is it possible"
    ]

    df["text_length"]=df["text"].str.len()
    df["exclamations"]=df["text"].str.count("!")
    df["Urgent_intent"]=df["text"].apply(lambda x: keyword_flag(x, urgent_words))
    df["complaint_intent"]= df["text"].apply(lambda x: keyword_flag(x, complaint_words))
    df["inquiry_intent"]= df["text"].apply(lambda x: keyword_flag(x, inquiry_words))

    return df

def keyword_flag(text, keywords):
    text = text.lower()
    return int(any(word in text for word in keywords))


#Creating text based numerical features for the model and urgency keyword flag
def build_preprocessor():
    cat_features = ["queue", "type"]
    num_features=["text_length","exclamations","Urgent_intent","complaint_intent","inquiry_intent"]

    preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),cat_features),
                                                   ("num", StandardScaler(), num_features),])
    
    return preprocessor
