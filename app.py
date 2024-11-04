from flask import Flask, make_response, request, abort, jsonify, render_template
from utils import make_soft_prediction, transform_input
import numpy as np

app = Flask(__name__)
app.debug = True

@app.route('/sentiment_score', methods=['POST'])
def get_sentiment_score():
    if not request.json or ('review' not in request.json):
        abort(400)

    inputs = request.get_json()
    inputs = inputs['review'].split(',')
    inputs = list(map(float, inputs))
    score = make_soft_prediction(inputs)

    response = {
        'review': inputs,
        'score': round(float(score),2)
    }

    print("response is: {}".format(response))

    return jsonify(response), 201
    
if __name__ == '__main__':
    app.run()

