from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import uvicorn
import index


# Define the data model
class Item(BaseModel):
    income : float
    collateral : float
    investments: float
    netcashflow : float
    cropyieldscore: float
    latepayments : int
    assessmentscore: int

# Load the model
with open('final_model.pkl', 'rb') as filea:
    model = pickle.load(file)

# Create the FastAPI application
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/recommend')
def predict_credit_score(item: Item):
    # Print the data received

    example_features = [
        float(item.income),
        float(item.collateral),
        float(item.investments),
        float(item.netcashflow),
        float(item.cropyieldscore),
        int(item.latepayments),
        int(item.assessmentscore)
    ]
   
    predicted_score = index.predict_credit_score(example_features)
    
    return predicted_score
    

# Run the application using uvicorn
# This should be in a separate file or under a __name__ == "__main__" condition
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)
    
# Run the application using uvicorn
# sudo uvicorn app:app --host 0.0.0.0 --port 8001