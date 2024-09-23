from fastapi import FastAPI, HTTPException
import backend as be
import os

app = FastAPI()

@app.post('/inference', response_model=be.InferenceResponse)
async def backend(request: be.UserRequest):
    try:
        inference_engine, model_name = be.get_inference_engine()

        model_path = be.TensorFlowInference.get_artifacts("hand-rekog", model_name , api_key=os.getenv('API_KEY')) 

        inference_engine.load_model(model_path)
        
        # Perform inference
        predictions = inference_engine.predict(request.image)
        print(f"Predictions: {predictions}")

        return {"output_bbox": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
