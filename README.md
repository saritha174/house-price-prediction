# 🏠 House Price Prediction using Machine Learning
This project builds a **Machine Learning system to predict house prices** using the California Housing dataset.
The project demonstrates a **complete ML pipeline** including:

- Data preprocessing
- Model training
- Model evaluation
- Prediction API
- Interactive Streamlit dashboard

# 🚀 Project Features

✔ Data preprocessing pipeline  
✔ Linear Regression model  
✔ Model evaluation metrics  
✔ Feature importance visualization  
✔ Interactive Streamlit dashboard  
✔ Prediction confidence interval  
✔ Dataset analytics

# 📂 Project Structure
house-price-prediction
│
├── app/
│ └── streamlit_app.py
│
├── data/
│ └── housing.csv
│
├── models/
│ └── model.pkl
│
├── notebooks/
│ └── eda.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── train_model.py
│ ├── evaluate.py
│ └── predict.py
│
├── requirements.txt
└── README.md

# ⚙️ Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Plotly
- Streamlit

# 📊 Dataset

Dataset used:
California Housing Dataset
Features include:
- Median Income
- Total Rooms
- Population
- Households
- Ocean Proximity
- Latitude & Longitude

# 🧠 Machine Learning Model
Algorithm used:
Linear Regression
The model predicts **median house value based on housing features**.

# ▶ Running the Project

### Install dependencies
pip install -r requirements.txt
### Train the model
python src/train_model.py
### Evaluate the model
python src/evaluate.py
### Run Streamlit dashboard
streamlit run app/streamlit_app.py

# 📈 Dashboard Features
The Streamlit dashboard provides:
- House price prediction
- Prediction gauge
- Feature importance chart
- Dataset visualization
- Downloadable prediction report

# 🎯 Future Improvements
- FastAPI prediction API
- Docker deployment
- Model monitoring
- Cloud deployment (AWS/GCP)


# 👩‍💻 Author
Saritha
AI / ML Engineer