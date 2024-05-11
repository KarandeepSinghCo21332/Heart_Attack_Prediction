# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import pickle

# Load the dataset
data = pd.read_csv("D:\Heart Attack Prediction\heart_dataset (1).csv")
data.head()
data.info()
Patients = data['target'].value_counts()
print("1: Higher Chances, 2: Low Chances")
print(Patients)

# Data
sizes = [526, 499]
labels = ['Higher Chances of Heart Attack', 'Lower Chances of Heart Attack']
colors = ['#ff9999','#66b3ff']

# Plot
plt.figure(figsize=(7,7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')  
plt.title('Distribution of Data')
plt.show()

# Calculate the number of people with high chances of heart attack above age 50
high_chance_above_50 = data[(data['age'] > 45) & (data['target'] == 1)].shape[0]

# Calculate the number of people with high chances of heart attack below or equal to age 50
high_chance_below_50 = data[(data['age'] <= 45) & (data['target'] == 1)].shape[0]

# Print the number of people with high and low chances of heart attack above and below age 50
print("Number of people with high chances of heart attack above age 50:", high_chance_above_50)
print("Number of people with high chances of heart attack below or equal to age 50:", high_chance_below_50)

# Data
labels = ['Above 50', 'Below or Equal to 50']
sizes = [370, 156]
colors = ['lightcoral', 'lightskyblue']

# Plotting the pie chart
plt.figure(figsize=(7,7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=0)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

# Title
plt.title('High Chances of Heart Attack by Age Group')

# Show the plot
plt.show()

# Split the dataset into features and target variable
X = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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

# Creating pickle file
pickle.dump(model,open('heart_attack_prediction.pkl','wb'))
rmodel=pickle.load(open('heart_attack_prediction.pkl','rb'))

# Printing Classification Report 
classification_report_xgb = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_report_xgb)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
classifiers = {
    "XG Boost": y_pred
}

sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
plt.title('Receiver Operating Characteristic Curve')

for name, y_pred in classifiers.items():
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_pred)
    plt.plot(false_positive_rate, true_positive_rate, label=f'{name} (AUC = {auc(false_positive_rate, true_positive_rate):.2f})')

plt.plot([0,1], ls='--')
plt.plot([0,0], [1,0], c='.5')
plt.plot([1,1], c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()