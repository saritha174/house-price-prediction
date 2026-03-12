import pickle
import numpy as np

#Load the Saved Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

#Create Input Data
new_house = [[
-122.23,   # longitude
37.88,     # latitude
41,        # housing_median_age
880,       # total_rooms
129,       # total_bedrooms
322,       # population
126,       # households
8.3252,    # median_income
1,         # INLAND
0,         # NEAR_BAY
0,         # NEAR_OCEAN
0,         # ISLAND
0          # <1H OCEAN
]]

#Predict Price
prediction = model.predict(new_house)

print("Predicted House Price:", prediction)