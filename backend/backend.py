from abc import ABC, abstractmethod
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import wandb
import io
from PIL import Image
import base64
import os
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

#app = FastAPI()

class UserRequest(BaseModel):
    image: str

class BoundingBox(BaseModel):
    endY: float
    startX: float
    startY: float
    endX: float

class InferenceResponse(BaseModel):
    output_bbox: list[BoundingBox]


class TensorFlowInference(ABC):
    @classmethod
    def get_artifacts(cls, project_name, model_name="latest", api_key=None):
        wandb.require("core")
        if api_key is not None:
            wandb.login(key=api_key)  # Use the app secret here
        else:
            wandb.login()  # Fallback to personal login

        run = wandb.init(project=project_name)
        downloaded_model_path = run.use_model(f"{model_name}:latest")
        return downloaded_model_path

    @classmethod
    def load_model(self, model_path):
        pass

    def preprocess_input(self, input_img):
        # Decode the base64-encoded image
        image_bytes = base64.b64decode(input_img)

         # Log the size of the received image bytes
        print(f"Received image bytes: {len(image_bytes)}")

        input_img = Image.open(io.BytesIO(image_bytes))
        input_img = input_img.resize((224, 224))
        input_img = self.img_to_array(input_img) / 255.0
        input_img = np.expand_dims(input_img, axis=0)
        input_img = input_img.astype(np.float32)

        # Log the shape of the preprocessed image
        print(f"Preprocessed image shape: {input_img.shape}")
        return input_img
    
    def postprocess_output(self, predictions, input_img):
        predictions = predictions.flatten()
        (h, w) = input_img.shape[1:3]

        (startX, startY, endX, endY) = predictions
        print(startX, startY, endX, endY)
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)

        bbox = [
            {"startX": startX, "startY": startY, "endX": endX, "endY": endY},
        ]
        # Return the predictions as a JSON response
        return bbox

    @abstractmethod
    def predict(self, input_img):
        pass

    def img_to_array(self, input_img):
        return np.array(input_img)


class MBM3TensorFlowInference(TensorFlowInference):
    def __init__(self):
        import tensorflow as tf
        self.tf = tf
        self.model = None

    def load_model(self, model_path):
        self.model = self.tf.keras.models.load_model(model_path)
    
    def predict(self, input_img):
        if self.model is None:
            raise ValueError("Model is not loaded")
        preprocessed_image = self.preprocess_input(input_img)
        raw_predictions = self.model.predict(preprocessed_image)
        return self.postprocess_output(raw_predictions, preprocessed_image)

class PiTPUInference(TensorFlowInference):
    def __init__(self):
        import tflite_runtime.interpreter as tflite
        self.tflite = tflite
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load_model(self, model_path):
        self.interpreter = self.tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                self.tflite.load_delegate(EDGETPU_SHARED_LIB)]
        )

        self.interpreter.allocate_tensors()
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_img):
        if self.interpreter is None:
            raise ValueError("Model is not loaded")

        preprocessed_image = self.preprocess_input(input_img)

        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)
        self.interpreter.invoke()
        raw_predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        return self.postprocess_output(raw_predictions, preprocessed_image)

def get_inference_engine():
    if platform.system() == 'Darwin':  # macOS
        print(f'"Using MBM3TensorFlowInference: {platform.system()}"')
        inference_engine = MBM3TensorFlowInference()
        model_name = "HAND-REKOG"
        return inference_engine, model_name
    elif platform.machine().startswith('aarch64'):  # Raspberry Pi
        inference_engine = PiTPUInference()
        model_name = "hand-rekog-tflite"
        return inference_engine, model_name
    else:
        raise NotImplementedError("Unsupported platform")

# @app.post('/inference', response_model=InferenceResponse)
# async def backend(request: UserRequest):
#     try:
#         inference_engine, model_name = get_inference_engine()

#         model_path = TensorFlowInference.get_artifacts("hand-rekog", model_name , api_key=os.getenv('API_KEY')) 
#         inference_engine.load_model(model_path)
        
#         # Perform inference
#         predictions = inference_engine.predict(request.image)
#         print(f"Predictions: {predictions}")

#         return {"output_bbox": predictions}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    