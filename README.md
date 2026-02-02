🎓 Student Dropout Prediction – Machine Learning App

Student Dropout Prediction is a Machine Learning–based web application that predicts whether a student is likely to pass or drop out based on personal, family, social, and academic factors.
The application uses a trained neural network model and provides predictions through an interactive Streamlit interface.

🌐 Live Demo

https://student-dropout-prediction-6zfw.onrender.com

🚀 Features

Student dropout / pass prediction

Machine Learning–based classification

Probability-based prediction output

Interactive and user-friendly Streamlit UI

Structured input form for student data

Scaled inputs using Min-Max normalization

Optimized performance using caching

Clean and professional layout suitable for demos    

🛠️ Tech Stack

Language: Python

Machine Learning: TensorFlow, Keras

Preprocessing: Scikit-learn (MinMaxScaler)

Web Framework: Streamlit

Data Handling: NumPy

Model Storage: .keras, .pkl

▶️ How to Run the Project Locally

1️⃣ Clone the repository

git clone https://github.com/Pranav2100/Student-Dropout-Prediction.git

cd Student-Dropout-Prediction

2️⃣ Create and activate a virtual environment

python -m venv venv

venv\Scripts\activate

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Run the Streamlit application

python -m streamlit run app.py

The app will open in your browser at:
http://localhost:8501

📊 Model Details

The model is trained to predict whether a student will pass (G3 ≥ 10).

Inputs include:

Personal & family background

Study habits

Social & health factors

Academic performance (G1, G2)

Predictions are based on a probability threshold of 0.80.

⚠️ Disclaimer

This project is created for educational and academic purposes only.

Predictions should not be used as the sole basis for academic or institutional decision-making.

👤 Author

Pranav Jagtap

GitHub: https://github.com/Pranav2100

LinkedIn: https://www.linkedin.com/in/pranav--jagtap

Email: pranavjagtap2151@gmail.com