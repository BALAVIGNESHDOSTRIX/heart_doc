from fastapi import FastAPI
import onnxruntime as ort 
import numpy as np
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

class HeartStrokeBase(BaseModel):
    ST_Slope: float
    ChestPainType: float 
    ExerciseAngina: float
    Cholesterol: float 
    MaxHR: float 
    Oldpeak: float
    Sex: int 
    FastingBS: float 
    Age: int 
    RestingBP: float  

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def heart_stroke_inference(data: HeartStrokeBase = None):
    
    print(data.dict())
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    try:
        data = data.dict().values()
        data = [x for x in data]
        data = np.array(data).astype(np.float32)
        data = [list(data)]
        model_path = 'HeartNet.onnx'
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: data})[0]
    
        bin_data = sigmoid(output)
        res = np.round(bin_data)
        predict_ = "yes" if int(res[0][0]) == 1 else "no"
        return {'type': "success", "value": predict_} 
    except Exception as e:
        return {'type': 'error', 'value': str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 