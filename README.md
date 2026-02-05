### Customer Support Ticket Priority Prediction

This project predicts the priority of customer support tickets using machine learning.

The goal is to automatically classify incoming tickets into priority levels so that urgent issues are handled faster.

---

### Problem Statement

Customer support teams receive a large number of tickets every day.

Manually identifying high priority and critical issues is slow and error prone.

This project builds an end to end machine learning pipeline to automatically predict ticket priority based on ticket metadata and text content.

---

### Dataset

The dataset contains customer support tickets with the following information:

- Queue or department
- Ticket subject
- Ticket body
- Ticket type
- Priority label

The dataset is highly imbalanced, with critical tickets being rare but very important

### Approach

I built a complete machine learning pipeline that includes:

- Data loading and cleaning
- Text feature extraction
- Categorical and numerical preprocessing
- Model training and evaluation
- API based deployment

Special care was taken to handle class imbalance and avoid data leakage.

---

### Models Trained

I trained and evaluated two models:

**Logistic Regression with class weighting**

- Used as a baseline
- Optimized to maximize recall for critical tickets

**Random Forest with balanced sampling**

- Final selected model
- Better balance between recall and precision across all classes

---

### Evaluation Metrics

Because missing critical tickets is costly, evaluation focused on recall and precision per class rather than accuracy alone.

### Logistic Regression (Balanced)

- Critical recall: 0.87
- Critical precision: 0.17
- Overall accuracy: 0.43

This model detects most critical tickets but produces many false escalations.

### Random Forest (Balanced)

- Critical recall: 0.86
- Critical precision: 0.76
- Overall accuracy: 0.80

This model achieves strong performance across all priority levels while still reliably detecting critical tickets.

---

### Final Model Choice

I selected the Random Forest model for deployment because it provides the best balance between detecting critical issues and reducing false alerts.

---

### Deployment

The final model is deployed locally using FastAPI.

The API exposes a `/predict` endpoint that accepts ticket details and returns a predicted priority.