import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

class SafetyScore:
    def __init__(self):
        self.HOTSPOT_RADIUS = 500
        self.MAX_RISK_DISTANCE = 4000
        self.HIGH_RISK_HOURS = [(16,22)]

        self.CRIME_WEIGHTS = {
        'pickpocketing': 0.17,
        'snatching': 0.16,
        'assault': 0.16,
        'abuse': 0.15,
        'molestation': 0.08,
        'vehicle_theft': 0.07,
        'murder': 0.06,
        'rape': 0.05,
        'robbery': 0.05,
        'burglary': 0.05
        }

        self.HOTSPOTS = {
        'Anand_Vihar': {'lat': 28.6469, 'lon': 77.3164, 'risk_weight': 0.9},
        'Seemapuri': {'lat': 28.6892, 'lon': 77.3238, 'risk_weight': 0.85},
        'Gokalpuri': {'lat': 28.7026, 'lon': 77.2789, 'risk_weight': 0.8},
        'Old_Delhi': {'lat': 28.6559, 'lon': 77.2293, 'risk_weight': 0.85},
        'New_Delhi': {'lat': 28.6430, 'lon': 77.2207, 'risk_weight': 0.75},
        'Kashmiri_Gate': {'lat': 28.6667, 'lon': 77.2287, 'risk_weight': 0.8},
        'Sarai_Kale_Khan': {'lat': 28.5918, 'lon': 77.2565, 'risk_weight': 0.7},
        'Jahangirpuri': {'lat': 28.7297, 'lon': 77.1667, 'risk_weight': 0.75},
        'Vishwavidyalaya': {'lat': 28.6880, 'lon': 77.2090, 'risk_weight': 0.6},
        'Vasant_Kunj': {'lat': 28.5400, 'lon': 77.1534, 'risk_weight': 0.5}
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
        for start_hour, end_hour in self.HIGH_RISK_HOURS:
            if start_hour <= hour < end_hour:
                return 1.0 - (abs(hour - (start_hour + end_hour)/2) / ((end_hour - start_hour)/2))
        return 0.3 
    

    def get_location_risk(self, lat, lon):
        max_risk = 0
        
        for hotspot, data in self.HOTSPOTS.items():
            distance = self.calculate_distance(lat, lon, data['lat'], data['lon'])
            
            if distance <= self.HOTSPOT_RADIUS:
                risk = data['risk_weight']
            elif distance <= self.MAX_RISK_DISTANCE:
                risk = data['risk_weight'] * (1 - (distance - self.HOTSPOT_RADIUS) / 
                                            (self.MAX_RISK_DISTANCE - self.HOTSPOT_RADIUS))
            else:
                risk = 0
                
            max_risk = max(max_risk, risk)
            
        return max_risk
    
    def calculate_safety_score(self, input_features):
        """
        Calculate safety score based on location, time, and other features
        input_features: [latitude, longitude, hour, is_crowded, has_streetlights, 
                        is_near_police, is_public_transport]
        """
        lat, lon, hour, is_crowded, has_streetlights, is_near_police, is_public_transport = input_features
        
        location_risk = self.get_location_risk(lat, lon)
        time_risk = self.get_time_risk_factor(hour)
        
        environmental_risk = 0.0
        if is_crowded:
            environmental_risk += 0.2  
        if not has_streetlights:
            environmental_risk += 0.3  
        if is_near_police:
            environmental_risk -= 0.3  
        if is_public_transport:
            environmental_risk += 0.1  
            
        total_risk = (0.4 * location_risk + 
                     0.3 * time_risk + 
                     0.3 * environmental_risk)

        safety_score = max(0, min(100, 100 * (1 - total_risk)))
        
        return safety_score


if __name__ == '__main__':
    safety_score = SafetyScore()
    # Input format: [latitude, longitude, hour, is_crowded, has_streetlights, is_near_police, is_public_transport]
    sample_input = [28.549932, 77.104259, 18, True, True, False, True]
    score = safety_score.calculate_safety_score(sample_input)
    print(score)