# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import pickle

# Load the dataset
data = pd.read_csv("D:\Heart Attack Prediction\heart_dataset (1).csv")

# Split the dataset into features and target variable
X = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
model = XGBClassifier( loss='log_loss', learning_rate=0.1, n_estimators=100, 
                      criterion='squared_error', random_state=None, max_leaf_nodes=10)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

pickle.dump(model,open('heart_attack_model.pkl','wb'))
rmodel=pickle.load(open('heart_attack_model.pkl','rb'))