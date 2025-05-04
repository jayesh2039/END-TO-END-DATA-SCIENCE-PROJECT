from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Sample synthetic data generation if file doesn't exist
def create_dummy_data():
    df = pd.DataFrame({
        'OverallQual': np.random.randint(1, 10, 100),
        'GrLivArea': np.random.randint(500, 4000, 100),
        'GarageCars': np.random.randint(0, 4, 100),
        'TotRmsAbvGrd': np.random.randint(2, 12, 100),
        'SalePrice': np.random.randint(50000, 500000, 100)
    })
    df.to_csv('train.csv', index=False)

# Preprocess data
def preprocess_data():
    if not os.path.exists('train.csv'):
        create_dummy_data()

    df = pd.read_csv('train.csv')
    X = df[['OverallQual', 'GrLivArea', 'GarageCars', 'TotRmsAbvGrd']]
    y = df['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save model
def train_model():
    X_train, X_test, y_train, y_test = preprocess_data()
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("Model trained. MSE:", mse)
    with open('house_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Load model or train if not found
try:
    with open('house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model file not found. Training new model...")
    train_model()
    with open('house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)

# HTML UI (Modern UI Style)
html_code = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6f8; }
        .container {
            max-width: 500px;
            margin: 60px auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        h2 { text-align: center; margin-bottom: 30px; }
        label { display: block; margin-top: 15px; font-weight: bold; }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .submit-btn {
            width: 100%;
            margin-top: 25px;
            padding: 12px;
            background: #007BFF;
            border: none;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
        }
        .submit-btn:hover {
            background: #0056b3;
        }
        .result {
            text-align: center;
            margin-top: 20px;
            font-size: 20px;
            color: #28a745;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>House Price Predictor</h2>
        <form method="POST">
            <label>Overall Quality (1–10):</label>
            <input type="number" name="OverallQual" required min="1" max="10">
            
            <label>Living Area (sq ft):</label>
            <input type="number" name="GrLivArea" required>
            
            <label>Garage Cars:</label>
            <input type="number" name="GarageCars" required>
            
            <label>Total Rooms Above Ground:</label>
            <input type="number" name="TotRmsAbvGrd" required>
            
            <button class="submit-btn" type="submit">Predict Price</button>
        </form>
        {% if predicted_price %}
            <div class="result">
                Predicted Price: ${{ predicted_price }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    predicted_price = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form['OverallQual']),
                float(request.form['GrLivArea']),
                float(request.form['GarageCars']),
                float(request.form['TotRmsAbvGrd'])
            ]
            prediction = model.predict([features])[0]
            predicted_price = round(prediction, 2)
        except Exception as e:
            predicted_price = f"Error: {str(e)}"

    return render_template_string(html_code, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
