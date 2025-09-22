# 🫀 Heart Disease Predictor

A machine learning web application that predicts the likelihood of heart disease in patients based on various medical parameters using Streamlit and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a machine learning model to predict heart disease risk in patients. The application uses various medical parameters such as age, blood pressure, cholesterol levels, and other clinical indicators to make predictions. The model is trained on a comprehensive heart disease dataset and deployed as a user-friendly web application.

### Key Components:
- **Machine Learning Model**: Trained using multiple algorithms (Logistic Regression, Decision Tree, Random Forest, SVM)
- **Web Interface**: Built with Streamlit for easy interaction
- **Data Processing**: Handles medical data preprocessing and feature engineering
- **Prediction System**: Provides binary classification (Heart Disease: Yes/No)

## ✨ Features

- 🔍 **Interactive Web Interface**: User-friendly form-based input system
- 🎯 **Multiple ML Algorithms**: Support for various classification models
- 📊 **Real-time Predictions**: Instant results with confidence indicators
- 🏥 **Medical Parameters**: Comprehensive set of clinical indicators
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🔒 **Input Validation**: Ensures data quality and range validation
- 📈 **Model Performance**: High accuracy prediction system

## 📊 Dataset

The project uses a heart disease dataset containing the following features:

| Feature | Description | Range/Values |
|---------|-------------|--------------|
| Age | Patient's age in years | 30-80 |
| Sex | Gender (0=Female, 1=Male) | 0, 1 |
| Chest Pain | Type of chest pain | 0-3 |
| Resting BP | Resting blood pressure (mm Hg) | 80-200 |
| Cholesterol | Serum cholesterol (mg/dl) | 100-600 |
| Fasting Blood Sugar | Fasting blood sugar >120 mg/dl | 0, 1 |
| Resting ECG | Resting electrocardiographic results | 0-2 |
| Max Heart Rate | Maximum heart rate achieved | 70-220 |
| Exercise Angina | Exercise-induced angina | 0, 1 |
| Old Peak | ST depression induced by exercise | 0.0-10.0 |
| ST Slope | Slope of peak exercise ST segment | 0-2 |
| Major Vessels | Number of major vessels colored | 0-3 |
| Target | Heart disease presence | 0, 1 |

**Dataset Statistics:**
- Total samples: 1,027
- Features: 13
- Target classes: 2 (No Heart Disease, Heart Disease)
- Missing values: None
- Data quality: Clean and preprocessed

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
# Clone the repository to your local machine
git clone https://github.com/yourusername/Heart-Disease-Predictor.git

# Navigate to the project directory
cd Heart-Disease-Predictor
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv heart_disease_env

# Activate the virtual environment
# On Windows:
heart_disease_env\Scripts\activate
# On macOS/Linux:
source heart_disease_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

**Required Packages:**
- `streamlit` - Web application framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning library
- `pickle` - Model serialization (built-in)

### Step 4: Verify Installation

```bash
# Check if all packages are installed correctly
pip list
```

## 💻 Usage

### Step 1: Train the Model (Optional)

If you want to retrain the model with new data:

```bash
# Run the Jupyter notebook
jupyter notebook HearDiseaseModel.ipynb
```

**Training Process:**
1. Load the dataset from `HeartDiseaseDataSet.csv`
2. Split data into training and testing sets (70% train, 30% test)
3. Train multiple models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
4. Evaluate model performance
5. Save the best model as `model.pkl`

### Step 2: Run the Web Application

```bash
# Start the Streamlit application
streamlit run main.py
```

**Application will be available at:** `http://localhost:8501`

### Step 3: Using the Application

1. **Open your web browser** and navigate to `http://localhost:8501`
2. **Fill in the patient information:**
   - Enter age (30-80 years)
   - Select gender (Male/Female)
   - Choose chest pain type (0-3)
   - Input resting blood pressure
   - Enter cholesterol level
   - Select fasting blood sugar status
   - Choose resting ECG results
   - Enter maximum heart rate
   - Select exercise angina status
   - Input old peak value
   - Choose ST slope
   - Select number of major vessels
3. **Click the "Predict" button**
4. **View the results:**
   - ✅ Green: No heart disease detected
   - ⚠️ Red: Heart disease risk detected

## 📁 Project Structure

```
Heart-Disease-Predictor/
│
├── 📄 README.md                 # Project documentation
├── 📄 requirements.txt          # Python dependencies
├── 🐍 main.py                   # Streamlit web application
├── 📓 HearDiseaseModel.ipynb    # Model training notebook
├── 📊 HeartDiseaseDataSet.csv   # Training dataset
├── 🤖 model.pkl                 # Trained model file
└── 📊 dataset.csv               # Additional dataset (if any)
```

### File Descriptions:

- **`main.py`**: Main Streamlit application with user interface
- **`HearDiseaseModel.ipynb`**: Jupyter notebook for model training and evaluation
- **`HeartDiseaseDataSet.csv`**: Primary dataset for training the model
- **`model.pkl`**: Serialized trained model (pickle format)
- **`requirements.txt`**: List of required Python packages
- **`README.md`**: Project documentation and usage instructions

## 🧠 Model Training

### Step 1: Data Loading and Preprocessing

```python
# Load the dataset
df = pd.read_csv("HeartDiseaseDataSet.csv")

# Separate features and target
X = df.drop("Target", axis=1)  # Features
y = df['Target']               # Target variable
```

### Step 2: Data Splitting

```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

### Step 3: Model Training

```python
# Define multiple models
models = {
    'LogisticRegression': LogisticRegression(),
    'Decision_Tree': DecisionTreeClassifier(),
    'Random_Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Train each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
```

### Step 4: Model Selection and Saving

```python
# Save the best performing model
import pickle
pickle.dump(best_model, open("model.pkl", 'wb'))
```

**Model Performance:**
- **SVM (Selected)**: 91.26% accuracy
- **Random Forest**: High accuracy with feature importance
- **Logistic Regression**: Good baseline performance
- **Decision Tree**: Interpretable but prone to overfitting

## 🌐 Web Application

### Step 1: Model Loading

```python
# Load the trained model
model = pickle.load(open("model.pkl", "rb"))
```

### Step 2: User Interface Creation

```python
# Create input form with two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 30, 80, 60)
    sex = st.selectbox("Sex", [0, 1])
    # ... other inputs

with col2:
    # ... additional inputs
```

### Step 3: Prediction Logic

```python
# Collect input data
input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                       restecg, thalach, exang, oldpeak,
                       slope, ca]])

# Make prediction
prediction = model.predict(input_data)[0]

# Display results
if prediction == 1:
    st.error("⚠️ Heart disease risk detected")
else:
    st.success("✅ No heart disease detected")
```

### Step 4: Result Display

The application provides:
- **Binary Classification**: Yes/No heart disease prediction
- **Visual Indicators**: Color-coded results (Green/Red)
- **User Feedback**: Clear success/error messages
- **Input Validation**: Range checking and data validation

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

### Step 1: Fork the Repository

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/Heart-Disease-Predictor.git
```

### Step 2: Create a Feature Branch

```bash
# Create a new branch for your feature
git checkout -b feature/your-feature-name
```

### Step 3: Make Changes

- Add new features
- Fix bugs
- Improve documentation
- Enhance UI/UX
- Optimize model performance

### Step 4: Test Your Changes

```bash
# Test the application
streamlit run main.py

# Run any additional tests
python -m pytest tests/
```

### Step 5: Submit a Pull Request

1. Commit your changes: `git commit -m "Add new feature"`
2. Push to your branch: `git push origin feature/your-feature-name`
3. Create a Pull Request on GitHub

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex code
- Update documentation for new features
- Test your changes thoroughly
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Heart Disease Predictor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 📞 Support

If you encounter any issues or have questions:

1. **Check the Issues**: Look through existing GitHub issues
2. **Create an Issue**: Report bugs or request features
3. **Contact**: Reach out to the maintainers
4. **Documentation**: Refer to this README and code comments

## 🙏 Acknowledgments

- **Dataset Source**: Heart Disease Dataset from UCI Machine Learning Repository
- **Libraries**: Streamlit, scikit-learn, pandas, numpy
- **Community**: Open source contributors and medical professionals
- **Inspiration**: Healthcare technology and machine learning applications

---

**⚠️ Medical Disclaimer**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

**🔬 Research Purpose**: This project demonstrates machine learning applications in healthcare and should be used responsibly with proper medical oversight.

---

Made with ❤️ for better healthcare through technology

