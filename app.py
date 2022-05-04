from flask import Flask, jsonify, request
import os
import pickle
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

model = pickle.load(open('ad_model','rb'))

@app.route('/api/v1/predict', methods=['GET'])
def predict():
    
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
    
    return jsonify({'predictions': prediction[0]})

@app.route('/api/v1/retrain', methods=['GET'])

def train():
    data = pd.read_csv('data/Advertising.csv', index_col=0)

    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.05,
                                                        random_state=42)

    model = Lasso(alpha=6000)
    model.fit(X_train, y_train)

    pickle.dump(model, open('ad_model.pkl', 'wb'))


app.run()