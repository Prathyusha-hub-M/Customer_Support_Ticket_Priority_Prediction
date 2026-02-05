from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_ticket

app= FastAPI()

class Ticket(BaseModel):
    queue : str
    subject: str
    body: str
    type: str

@app.post("/predict")
def predict(ticket: Ticket):
    data = ticket.model_dump()
    return predict_ticket(data)
