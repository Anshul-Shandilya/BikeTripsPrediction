import numpy as np
import pickle
from flask import Flask, request, render_template

# Create application
app = Flask(__name__)

# Load machine learning model
model_sf = pickle.load(open('sf.pkl', 'rb'))
model_mv = pickle.load(open('mv.pkl', 'rb'))
model_sj = pickle.load(open('sj.pkl', 'rb'))
model_re = pickle.load(open('re.pkl', 'rb'))


# Bind home function to URL
@app.route('/')
def home():
    return render_template('app_new.html')


# Bind predict function to URL
@app.route('/ predict', methods=['POST'])
def predict():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]

    # Remove first element from features and store it in a variable
    city = features.pop(0)

    if city == 0:
        prediction = model_sf.predict([features])
    elif city == 1:
        prediction = model_sj.predict([features])
    elif city == 2:
        prediction = model_re.predict([features])
    else:
        prediction = model_mv.predict([features])

    # Check the output values and retrieve the result with html tag based on the value
    return render_template('app_new.html', result=prediction)


app.run()
