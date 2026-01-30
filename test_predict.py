from src.predict import predict_ticket

sample_ticket = {
    "queue": "Technical Support",
    "subject": "System down",
    "body": "Our production system is not working and users are blocked",
    "type": "Incident"
}

print(predict_ticket(sample_ticket))
