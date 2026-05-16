Heart Disease Prediction System 🫀

A Machine Learning web application that predicts the risk of heart disease based on health parameters using a Random Forest Classifier.
This project is developed using Python, Scikit-learn, and Streamlit to provide an interactive and user-friendly prediction system.

The application analyzes patient health details and predicts whether the person has a high risk or low risk of heart disease.

Live Demo link: https://a-random-forest-classfier-6.streamlit.app/

🚀 Features
Predicts heart disease risk instantly
Interactive Streamlit web application
Uses Random Forest Machine Learning algorithm
Displays prediction confidence score
Simple and beginner-friendly UI
Real-time health analysis
🛠️ Technologies Used
Python
Streamlit
Pandas
NumPy
Scikit-learn
Joblib
Matplotlib
Seaborn
📂 Project Structure
├── output.py                   # Streamlit application
├── input.ipynb                 # Model training notebook
├── heart_disease_dataset.csv   # Dataset
├── heart_rf_model.pkl          # Trained Random Forest model
├── heart_scaler.pkl            # Feature scaler
├── sex_encoder.pkl             # Label encoder for gender
├── requirements.txt            # Required libraries
└── README.md                   # Project documentation
📊 Dataset Features

The model uses the following health parameters:

Age
Sex
Blood Pressure
Cholesterol
Max Heart Rate
🧠 Machine Learning Model

This project uses the Random Forest Classifier algorithm for heart disease prediction.

Workflow:
Data preprocessing
Feature scaling
Encoding categorical values
Model training
Prediction generation
⚙️ Installation
1. Clone the Repository
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
2. Install Required Libraries
pip install -r requirements.txt
▶️ Run the Application
streamlit run output.py

The application will open automatically in your browser.

📈 Application Workflow
User enters health details:
Age
Sex
Blood Pressure
Cholesterol
Max Heart Rate
The system preprocesses the input data
The trained model predicts:
High Risk of Heart Disease
Low Risk of Heart Disease
Prediction confidence score is displayed

The Streamlit application implementation is available in output.py.

📦 Requirements

Required libraries are listed in requirements.txt.

Main libraries include:

pandas
streamlit
joblib
scikit-learn
matplotlib
seaborn
🎯 Future Improvements
Add more medical parameters
Improve model accuracy
Add data visualization dashboard
Deploy on cloud platforms
Add PDF report generation

👨‍💻 Author
Anand

📜 License

This project is open-source and available under the MIT License.
