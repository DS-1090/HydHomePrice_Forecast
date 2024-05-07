from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('forest_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('webpage.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    print(prediction)
    print(final_features)
    return render_template('webpage.html', prediction_text='The predicted price of the house is {} Lakhs'.format(output))

if __name__ == "__main__":
    app.run(debug=True)