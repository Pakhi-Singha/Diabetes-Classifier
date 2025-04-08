## Step 1: Install required libraries
!pip install xgboost scikit-learn pandas matplotlib ipywidgets lightgbm catboost

## Step 2: Import libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

## Step 3: Load the dataset (Assuming it's already uploaded to Colab)
file_path = '/content/Diabetes Classification.csv'  # Update if needed
df = pd.read_csv(file_path)

## Step 4: Data Preprocessing
# Drop unnecessary column if exists
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

# Encode categorical column (Gender)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Separate features and target
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

## Step 5: Train Models & Evaluate Performance

# 1. XGBoost Model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, max_depth=5, learning_rate=0.1, n_estimators=300, subsample=0.8, colsample_bytree=0.8)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

# 2. LightGBM Model
lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=5, learning_rate=0.1, n_estimators=300, subsample=0.8, colsample_bytree=0.8)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_acc = accuracy_score(y_test, lgb_pred)

# 3. CatBoost Model
cat_model = cb.CatBoostClassifier(iterations=300, depth=5, learning_rate=0.1, verbose=0)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
cat_acc = accuracy_score(y_test, cat_pred)

# 4. Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# 5. Support Vector Machine (SVM) Model
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# 6. Neural Network (MLP) Model
nn_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)
nn_acc = accuracy_score(y_test, nn_pred)

# Print Accuracy for all models
print(f"XGBoost Model Accuracy: {xgb_acc:.4f}")
print(f"LightGBM Model Accuracy: {lgb_acc:.4f}")
print(f"CatBoost Model Accuracy: {cat_acc:.4f}")
print(f"Random Forest Model Accuracy: {rf_acc:.4f}")
print(f"SVM Model Accuracy: {svm_acc:.4f}")
print(f"Neural Network Model Accuracy: {nn_acc:.4f}")

## Step 6: Model Evaluation - Select Best Model (Example: XGBoost) for detailed evaluation
best_model = xgb_model  # You can select the model with the highest accuracy
print("\nClassification Report for Best Model:\n", classification_report(y_test, best_model.predict(X_test)))
print("\nConfusion Matrix for Best Model:\n", confusion_matrix(y_test, best_model.predict(X_test)))

## Step 7: User Input for Prediction using Widgets
def predict_user_input():
    age = widgets.IntText(description="Age:")
    gender = widgets.Dropdown(options=[("Male", 1), ("Female", 0)], description="Gender:")
    bmi = widgets.FloatText(description="BMI:")
    chol = widgets.FloatText(description="Cholesterol:")
    tg = widgets.FloatText(description="Triglycerides:")
    hdl = widgets.FloatText(description="HDL:")
    ldl = widgets.FloatText(description="LDL:")
    cr = widgets.FloatText(description="Creatinine:")
    bun = widgets.FloatText(description="BUN:")

    def on_click(b):
        user_data = np.array([[age.value, gender.value, bmi.value, chol.value, tg.value, hdl.value, ldl.value, cr.value, bun.value]])
        user_data = scaler.transform(user_data)
        prediction = best_model.predict(user_data)
        print("\nPredicted Diagnosis:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")

    button = widgets.Button(description="Predict")
    button.on_click(on_click)
    display(age, gender, bmi, chol, tg, hdl, ldl, cr, bun, button)

# Run user input function after training
predict_user_input()
