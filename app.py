from flask import Flask, make_response, request, abort, jsonify
import numpy as np
from train.main import SafetyScore

app = Flask(__name__)
app.debug = True

def transform_input(inputs):
    """
    Transform the input array into the format expected by SafetyScore
    Input format: [hour, minute, latitude, longitude, day, month]
    Output format: [latitude, longitude, hour, is_crowded, has_streetlights, is_near_police, is_public_transport]
    """
    hour = inputs[0]
    latitude = inputs[2]
    longitude = inputs[3]
    
    # Default values for unused parameters
    is_crowded = True  # Default assumption
    has_streetlights = True  # Default assumption
    is_near_police = False  # Default assumption
    is_public_transport = True  # Default assumption
    
    return [latitude, longitude, hour, is_crowded, has_streetlights, is_near_police, is_public_transport]

@app.route('/sentiment_score', methods=['POST'])
def get_sentiment_score():
    if not request.json or ('review' not in request.json):
        abort(400)

    try:
        # Get input data
        inputs = request.get_json()
        inputs = inputs['review'].split(',')
        inputs = list(map(float, inputs))
        
        if len(inputs) != 6:  # Check if we have all required inputs
            abort(400, description="Input must contain exactly 6 values: hour, minute, latitude, longitude, day, month")
        
        # Transform inputs to required format
        transformed_inputs = transform_input(inputs)
        
        # Calculate safety score
        safety_scorer = SafetyScore()
        score = safety_scorer.calculate_safety_score(transformed_inputs)

        response = {
            'review': inputs,
            'score': round(float(score), 2)
        }

        print("response is: {}".format(response))
        return jsonify(response), 201

    except ValueError as e:
        abort(400, description="Invalid input format. All values must be numbers.")
    except Exception as e:
        abort(500, description=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)