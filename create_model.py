# create_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# ✅ Sample data (Score & Age vs. Passed)
data = pd.DataFrame({
    "Score": [95, 40, 65, 20, 88, 35],
    "Age": [22, 25, 19, 23, 21, 24],
    "Passed": [1, 0, 1, 0, 1, 0]
})

X = data[["Score", "Age"]]
y = data["Passed"]

# ✅ Train model
model = LogisticRegression()
model.fit(X, y)

# ✅ Save clean model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")

