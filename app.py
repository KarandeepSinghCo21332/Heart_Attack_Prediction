from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('heart_attack_prediction.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])

    # Create a numpy array with the input data
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Make the prediction
    prediction = model.predict(input_data)[0]

    # Determine the result based on the prediction
    if prediction == 0:
        result = "Congratulations, You are safe!! NO HEART ATTACK ❤️"
    else:
        result = "Consult your doctor ASAP, You have high chances of HEART ATTACK 💔"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)