from datasets import load_dataset

def load_tickets():
    #get the dataset from huggingface and load as a dictionary(Usually the dataset is in form of a dictionary with keys being train, test, validation datsets) to dataset variable
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
    #converting the train part of the datasset to pandas dataframe
    df = dataset["train"].to_pandas()
    return df