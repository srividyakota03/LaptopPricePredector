# Laptop Price Predictor

Welcome to the **Laptop Price Predictor** — your handy companion to estimate the price of a laptop based on its features. Whether you’re a buyer trying to spot a good deal or a seller wanting to price your machine fairly, this project aims to make the guessing game a lot easier and data-driven!

---

## What This Project Does

Using a dataset of laptops and their specifications, this project builds a machine learning model that learns the relationship between hardware features (like RAM, screen size, CPU, storage) and their market prices.  
Once trained, the model can predict the estimated price of a laptop given its specs. This prediction helps you understand if a laptop is priced too low, too high, or just right.

---

## Key Features

- Predict laptop price based on popular features like brand, RAM, storage, screen size, and more.
- Interactive web app built with Streamlit for easy user input and instant price estimation.
- Clean data preprocessing and feature engineering for accurate modeling.
- Model powered by Random Forest Regressor, a strong and reliable machine learning algorithm.
- Label encoding for categorical variables to make machine learning friendly.
- Intuitive UI with a sidebar for specs input and instant display of predicted price.

---

## Tech Stack

| Technology          | Purpose                      |
|---------------------|------------------------------|
| Python              | Core programming language     |
| Pandas & NumPy      | Data manipulation & processing|
| Scikit-learn        | Machine learning modeling     |
| Streamlit           | Web app interface             |
| Pickle              | Saving/loading model & encoders|

---

## 📂 Project Structure
```
Laptop-Price-Predictor/
├── laptop_prices.csv          # Dataset file containing laptop specs and prices
├── model.pkl                  # Trained Random Forest regression model
├── label_encoders.pkl         # Encoders for categorical features
├── train_model.py             # Script to preprocess data and train the model
├── app.py                     # Streamlit web app for laptop price prediction
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

