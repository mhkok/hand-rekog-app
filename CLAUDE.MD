# CLAUDE.MD - Hand Recognition App

## Project Overview
Hand recognition application that uses camera input to detect and recognize hand gestures. The system is designed to run on a Raspberry Pi cluster with Google Coral TPU acceleration for efficient ML inference.

## Architecture

### Tech Stack
- **Frontend**: Streamlit (web-based camera interface)
- **Backend**: FastAPI (REST API for ML inference)
- **ML Framework**: TensorFlow (optimized for Coral TPU)
- **Infrastructure**: Kubernetes (k3s) on Raspberry Pi cluster
- **Hardware**: 3-node Raspberry Pi cluster with Coral TPU accelerator

### System Design
```
[Streamlit Frontend] --HTTP--> [FastAPI Backend] ---> [TensorFlow Model] ---> [Coral TPU]
      (Local)                   (k8s/Pi Cluster)          (Inference)
```

## Project Structure

```
hand-rekog-app/
├── streamlit.py              # Frontend: Streamlit camera interface
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── backend.py           # Inference engine and core logic
│   ├── requirements.txt     # Python dependencies (general)
│   ├── requirements.backend.arm64.txt  # ARM64-specific dependencies
│   ├── Containerfile        # Container build specification
│   └── k8s/
│       └── backend.yaml     # Kubernetes deployment manifest
└── coral/
    └── requirements.coral.arm64.txt  # Coral TPU dependencies
```

## Key Components

### Frontend (`streamlit.py:1`)
- **Purpose**: Captures camera input and displays hand detection results
- **Key Features**:
  - Camera input via Streamlit's `camera_input` widget
  - Base64 image encoding for API transmission
  - Bounding box visualization using OpenCV
  - Real-time inference results display
- **API Endpoint**: `POST http://localhost:8000/inference`

### Backend (`backend/main.py:1`)
- **Framework**: FastAPI
- **Main Endpoint**: `/inference` (POST)
- **Flow**:
  1. Receives base64-encoded image
  2. Initializes inference engine (TensorFlow or Coral-optimized)
  3. Loads model from artifacts (requires `API_KEY` environment variable)
  4. Performs inference
  5. Returns bounding box predictions

### Inference Engine (`backend/backend.py`)
- Handles model loading and prediction
- Supports multiple inference backends (likely TensorFlow standard and Coral-optimized)
- Model management via `TensorFlowInference.get_artifacts()`

## Development Setup

### Prerequisites
- Python environment (pyenv recommended)
- For backend: Access to Kubernetes cluster (k3s) or local development setup
- For TPU inference: Coral TPU accelerator

### Running Locally

**Frontend**:
```bash
# From project root
streamlit run streamlit.py
```

**Backend**:
```bash
# From backend/ directory
cd backend/
uvicorn main:app
```

### Deploying to Kubernetes
```bash
# Apply backend deployment
kubectl apply -f backend/k8s/backend.yaml
```

## Environment Variables

### Required
- `API_KEY`: Authentication key for model artifact retrieval

### Configuration
- Backend runs on port 8000 by default
- Frontend connects to `http://localhost:8000/inference`

## Data Flow

1. **Image Capture**: User takes photo via Streamlit camera input
2. **Encoding**: Image converted to base64 JPEG format
3. **Request**: Base64 string sent to `/inference` endpoint
4. **Processing**: Backend loads model and runs inference on Coral TPU
5. **Response**: Returns array of bounding boxes with coordinates:
   ```json
   {
     "output_bbox": [
       {"startX": 100, "startY": 150, "endX": 200, "endY": 250}
     ]
   }
   ```
6. **Visualization**: Frontend draws bounding boxes on image using OpenCV

## Infrastructure Details

### Kubernetes Cluster
- **Platform**: k3s (lightweight Kubernetes)
- **Nodes**: 3 Raspberry Pi devices
- **Setup Tool**: k3sup (https://github.com/alexellis/k3sup)
- **Hardware**: Coral TPU accelerator (https://coral.ai/docs/accelerator/get-started)

### Deployment Strategy
- Containerized backend (see `Containerfile`)
- Kubernetes manifests in `backend/k8s/`
- ARM64-specific dependencies for Pi architecture

## Model Information

- **Project**: "hand-rekog"
- **Format**: TensorFlow model optimized for Coral TPU
- **Task**: Hand detection with bounding box output
- **Artifacts**: Retrieved via API key authentication

## Important Patterns

### Error Handling
- FastAPI uses HTTPException for error responses (status 500)
- Frontend displays status codes and error messages to user

### Image Processing
- Images processed as JPEG format
- Coordinate system: (startX, startY, endX, endY) for bounding boxes
- OpenCV used for drawing (BGR color space, converted to RGB for display)

### API Contract
- **Request**: `{"image": "<base64-encoded-jpeg>"}`
- **Response**: `{"output_bbox": [{"startX": int, "startY": int, "endX": int, "endY": int}]}`

## Development Notes

### When Adding Features
- Frontend changes: Modify `streamlit.py`
- API changes: Update `backend/main.py` (endpoint) and `backend/backend.py` (logic)
- New models: Update model name in `main.py:12`
- Deployment: Update `backend/k8s/backend.yaml` for k8s changes

### Dependencies
- Use `requirements.txt` for general dependencies
- Use `requirements.backend.arm64.txt` for ARM64-specific packages
- Use `requirements.coral.arm64.txt` for Coral TPU-specific packages

### Testing
- Frontend: Run locally with `streamlit run streamlit.py`
- Backend: Test with `uvicorn main:app --reload` for hot-reload
- Integration: Ensure both services running and accessible on correct ports

## Common Tasks

### Update Model
1. Modify model name in `backend/main.py:12`
2. Ensure model artifacts accessible via API key
3. Redeploy backend if running on k8s

### Change Inference Endpoint
1. Update URL in `streamlit.py:21`
2. Match with backend service URL (localhost:8000 or k8s service)

### Add New Endpoints
1. Define response model in `backend/backend.py`
2. Add endpoint decorator in `backend/main.py`
3. Update frontend to call new endpoint

## References
- Coral TPU: https://coral.ai/docs/accelerator/get-started
- k3sup: https://github.com/alexellis/k3sup
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://streamlit.io/
