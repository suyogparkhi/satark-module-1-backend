from flask import Flask, request, abort, jsonify
import numpy as np

class SafetyScore:
    def __init__(self):
        # Adjusted radius parameters for more gradual distance effects
        self.HOTSPOT_RADIUS = 300  # Reduced from 500
        self.MAX_RISK_DISTANCE = 3000  # Reduced from 4000
        
        # Multiple time windows with different risk levels
        self.RISK_HOURS = {
            (22, 5): 1.0,    # Highest risk (10 PM - 5 AM)
            (19, 22): 0.8,   # High risk (7 PM - 10 PM)
            (16, 19): 0.6,   # Moderate risk (4 PM - 7 PM)
            (5, 7): 0.5,     # Early morning risk (5 AM - 7 AM)
            (7, 16): 0.2     # Daytime (7 AM - 4 PM)
        }

        # Updated hotspots with more precise risk weights
        self.HOTSPOTS = {
            'Anand_Vihar': {'lat': 28.6469, 'lon': 77.3164, 'risk_weight': 0.95},
            'Seemapuri': {'lat': 28.6892, 'lon': 77.3238, 'risk_weight': 0.90},
            'Gokalpuri': {'lat': 28.7026, 'lon': 77.2789, 'risk_weight': 0.85},
            'Old_Delhi': {'lat': 28.6559, 'lon': 77.2293, 'risk_weight': 0.80},
            'Kashmiri_Gate': {'lat': 28.6667, 'lon': 77.2287, 'risk_weight': 0.75},
            'Jahangirpuri': {'lat': 28.7297, 'lon': 77.1667, 'risk_weight': 0.75},
            'New_Delhi': {'lat': 28.6430, 'lon': 77.2207, 'risk_weight': 0.70},
            'Sarai_Kale_Khan': {'lat': 28.5918, 'lon': 77.2565, 'risk_weight': 0.65},
            'Vishwavidyalaya': {'lat': 28.6880, 'lon': 77.2090, 'risk_weight': 0.50}
        }

        # Safe zones with safety boosters
        self.SAFE_ZONES = {
            'IGI_Airport': {'lat': 28.5562, 'lon': 77.1000, 'radius': 2000, 'safety_boost': 0.8},
            'Diplomatic_Area': {'lat': 28.6007, 'lon': 77.1833, 'radius': 1500, 'safety_boost': 0.7},
            'Delhi_Cantonment': {'lat': 28.5917, 'lon': 77.1504, 'radius': 3000, 'safety_boost': 0.6},
            'Supreme_Court': {'lat': 28.6234, 'lon': 77.2419, 'radius': 1000, 'safety_boost': 0.5},
            'Moti_Bagh': {'lat': 28.5593472, 'lon': 77.1746443, 'radius': 1500, 'safety_boost': 0.65}

        }

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = (np.sin(delta_phi/2) * np.sin(delta_phi/2) +
             np.cos(phi1) * np.cos(phi2) *
             np.sin(delta_lambda/2) * np.sin(delta_lambda/2))
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def get_time_risk_factor(self, hour):
        hour = hour % 24
        
        # Find applicable time window
        for (start, end), risk in self.RISK_HOURS.items():
            if start <= hour < end or (start > end and (hour >= start or hour < end)):
                return risk
                
        # Default risk for any uncovered hours
        return 0.3

    def get_location_risk(self, lat, lon):
        max_risk = 0
        
        # Calculate risk from hotspots
        for hotspot, data in self.HOTSPOTS.items():
            distance = self.calculate_distance(lat, lon, data['lat'], data['lon'])
            
            if distance <= self.HOTSPOT_RADIUS:
                risk = data['risk_weight']
            elif distance <= self.MAX_RISK_DISTANCE:
                # Exponential decay of risk with distance
                risk = data['risk_weight'] * np.exp(-distance / self.MAX_RISK_DISTANCE)
            else:
                risk = 0
                
            max_risk = max(max_risk, risk)
            
        return max_risk

    def get_safety_boost(self, lat, lon):
        max_boost = 0
        
        # Check if location is within any safe zone
        for zone, data in self.SAFE_ZONES.items():
            distance = self.calculate_distance(lat, lon, data['lat'], data['lon'])
            
            if distance <= data['radius']:
                # Linear decrease in safety boost with distance
                boost = data['safety_boost'] * (1 - distance / data['radius'])
                max_boost = max(max_boost, boost)
                
        return max_boost

    def calculate_safety_score(self, lat, lon, hour):
        """
        Calculate safety score based on location and time
        Returns a score between 0 (unsafe) and 100 (safe)
        """
        location_risk = self.get_location_risk(lat, lon)
        time_risk = self.get_time_risk_factor(hour)
        safety_boost = self.get_safety_boost(lat, lon)
        
        # Calculate base risk (weighted average of location and time risks)
        base_risk = 0.6 * location_risk + 0.4 * time_risk
        
        # Apply safety boost (reduces risk)
        total_risk = base_risk * (1 - safety_boost)
        
        # Convert to safety score (0-100)
        safety_score = max(0, min(100, 100 * (1 - total_risk)))
        
        return safety_score

app = Flask(__name__)
app.debug = True

@app.route('/sentiment_score', methods=['POST'])
def get_sentiment_score():
    if not request.json or ('review' not in request.json):
        abort(400)

    try:
        inputs = request.get_json()['review'].split(',')
        inputs = list(map(float, inputs))
        
        if len(inputs) != 6:
            abort(400, description="Input must contain exactly 6 values: hour, minute, latitude, longitude, day, month")
        
        hour = inputs[0]
        latitude = inputs[2]
        longitude = inputs[3]
        
        safety_scorer = SafetyScore()
        score = safety_scorer.calculate_safety_score(latitude, longitude, hour)

        return jsonify({
            'review': inputs,
            'score': round(float(score), 2)
        }), 201

    except ValueError as e:
        abort(400, description="Invalid input format. All values must be numbers.")
    except Exception as e:
        abort(500, description=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)