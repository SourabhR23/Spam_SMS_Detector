# Import the essential libraries
from flask import Flask, render_template, url_for, request
import pickle
import preprocessing


# Load the Multinomial Naive Bayes model and CountVectorizer model from disk
cv = pickle.load(open('cv-transform.pkl', 'rb'))
model = pickle.load(open('mnb-model-spam-analysis.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['POST'])
def predict():
    message = request.form['message']
    text = [message]
    data = preprocessing.process(text)
    vec = cv.transform(data)
    prediction = model.predict(vec)
    return render_template('result.html', prediction = prediction)


if __name__ == '__main__':
    app.run(debug = True)


