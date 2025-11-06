# ✈️ Flight Price Prediction using Machine Learning 

## 📘 Project Overview
This project is an interactive **web application built using Streamlit** that predicts **domestic flight ticket prices in India**.  
The machine learning model analyzes multiple travel parameters such as **airline, source, destination, total stops, departure time, and arrival time** to estimate the ticket cost for Indian flight routes.

---

## 🚀 Key Features
- **Multiple ML Models:** Implemented and compared **Random Forest**, **Decision Tree**, and **K-Nearest Neighbors** to identify the most accurate predictor.  
- **Data Preprocessing:**  
  - Handled missing values using `SimpleImputer`.  
  - Encoded categorical features using `OneHotEncoder`.  
  - Scaled numerical features with `StandardScaler`.  
- **Feature Engineering:** Extracted useful patterns from **departure and arrival times** to enhance model performance.  
- **Interactive Web App:**  
  Created a user-friendly **Streamlit interface** where users can input flight details and instantly get price predictions.  
- **Model Evaluation:**  
  Evaluated models using performance metrics such as **R² Score**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.

---

## 🧠 Machine Learning Workflow
1. **Data Preprocessing:** Cleaning, handling null values, and encoding categorical variables.  
2. **Feature Engineering:** Extracting new features from time-based data.  
3. **Model Training:** Training multiple supervised learning models.  
4. **Model Evaluation:** Comparing performance and selecting the best model.  
5. **Deployment:** Building a Streamlit interface for real-time predictions.

---

## 🧰 Tech Stack
- **Programming Language:** Python  
- **Libraries Used:**  
  - `pandas`, `numpy` – Data manipulation and analysis  
  - `scikit-learn` – Machine learning model training and evaluation  
  - `joblib` – Model serialization  
  - `streamlit` – Web app development  
