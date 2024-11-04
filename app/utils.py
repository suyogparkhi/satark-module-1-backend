import pickle
import numpy as np

grid_rf = pickle.load(open('./grid_rf.pkl', 'rb'))
scaler  = pickle.load(open('./scaler.pkl', 'rb'))


def transform_input(input):
	return scaler.transform([input])

def make_hard_prediction(input):
    return grid_rf.predict(transform_input(input))

def make_soft_prediction(input):
    return grid_rf.predict_proba(transform_input(input))[0,1]