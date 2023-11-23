import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from Requirement import HandleOutlier, AgeBinner, AgeTransformer, ColumnDropper

# Load All Data Cleaned
data = pd.read_csv('D:\\david\\OneDrive\\Documents\\Future\\Project\\Application\\Education\\Purwadhika\\Capstone\\ApartmentData\\Notebook\\data_daegu_apartment_preparation_cleaned.csv')
data.to_dict(orient='records')

app = FastAPI(
    title="Apartment Prediction API", 
    version="v1.0.0"
)

# Definisikan struktur data input menggunakan Pydantic
class PricePredictions(BaseModel):
    HallwayType: str
    TimeToSubway: str
    SubwayStation: str
    N_FacilitiesNearBy_ETC: float
    N_FacilitiesNearBy_PublicOffice: float
    N_SchoolNearBy_University: float
    N_Parkinglot_Basement: float
    YearBuilt: int
    N_FacilitiesInApt: int
    Size_sqf: int
    SalePrice: int

# Define a Python class to create a list to reformat the data
class Item(BaseModel):
    data: List[PricePredictions]

# define data
payload = {
    "data": [
    {
    'HallwayType': 'terraced',
    'TimeToSubway': '0-5min',
    'SubwayStation': 'Banwoldang',
    'N_FacilitiesNearBy(ETC)': 0.0,
    'N_FacilitiesNearBy(PublicOffice)': 4.0,
    'N_SchoolNearBy(University)': 1.0,
    'N_Parkinglot(Basement)': 605.0,
    'YearBuilt': 2007,
    'N_FacilitiesInApt': 5,
    'Size(sqf)': 1334,
    'SalePrice': 357522
    },
    ],
}

# Loading the saved model
model = pickle.load(open('D:\\david\\OneDrive\\Documents\\Future\\Project\\Application\\Education\\Purwadhika\\Capstone\\ApartmentData\\Notebook\\gbr_finalmodel.sav', 'rb'))

class Item(BaseModel):
    data: List[PricePredictions]

# Create a POST endpoint to make prediction
@app.post('/')
async def diabetes_prediction(parameters: Item):
    # Get inputs
    req = parameters.dict()['data']

    # Convert input into Pandas DataFrame
    data = pd.DataFrame(req)

    # Make the predictions
    res = model.predict(data).tolist()
    
    return {"Request": req, "Response": res}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
