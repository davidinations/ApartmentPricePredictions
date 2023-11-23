import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pycaret.regression import *
from Requirement import HandleOutlier, AgeBinner, AgeTransformer, ColumnDropper

# Load All Data Cleaned

app = FastAPI(
    title="Apartment Prediction API", 
    version="v1.0.0"
)

# Definisikan struktur data input menggunakan Pydantic
class PricePredictions(BaseModel):
    HallwayType: object
    TimeToSubway: object
    SubwayStation: object
    N_FacilitiesNearBy_ETC: float
    N_FacilitiesNearBy_PublicOffice: float
    N_SchoolNearBy_University: float
    N_Parkinglot_Basement: float
    YearBuilt: int
    N_FacilitiesInApt: int
    Size(sqf): int
    SalePrice: int

# Define a Python class to create a list to reformat the data
class Item(BaseModel):
    data: List[PricePredictions]

# # define data
# payload = {
#     "data": [
#     {
#     'HallwayType': 'terraced',
#     'TimeToSubway': '0-5min',
#     'SubwayStation': 'Banwoldang',
#     'N_FacilitiesNearBy(ETC)': 0.0,
#     'N_FacilitiesNearBy(PublicOffice)': 4.0,
#     'N_SchoolNearBy(University)': 1.0,
#     'N_Parkinglot(Basement)': 605.0,
#     'YearBuilt': 2007,
#     'N_FacilitiesInApt': 5,
#     'Size(sqf)': 1334,
#     'SalePrice': 357522
#     },
#     ],
# }

# Loading the saved model
model = load_model('model/final_model') 

# Create a POST endpoint to make prediction
@app.post('/prediction')
async def diabetes_prediction(parameters: Item):
    # Get inputs
    req = parameters.dict()['data']

    # Convert input into Pandas DataFrame
    data = pd.DataFrame(req)

    # Make the predictions
    res = predict_model(estimator=model, data=data).tolist()
    
    return {"Request": req, "Response": res}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
