# CLAUDE.MD - Project Context for AI Assistant

## Project Overview

**hand-rekog-app** is a web-based application for recognizing hand gestures through a camera, designed to run on a Raspberry Pi with Coral TPU acceleration.

## Architecture

The application follows a client-server architecture:

### Frontend
- **Technology**: Streamlit
- **Location**: Root directory (`streamlit.py`)
- **Communication**: Connects to backend via `/inference` endpoint
- **Running**: Currently runs locally on developer's machine

### Backend
- **Technology**: FastAPI (uvicorn)
- **Location**: `backend/` directory
- **Main Files**:
  - `backend/main.py` - FastAPI application with inference endpoint
  - `backend/backend.py` - Core inference engine logic
- **API Endpoints**:
  - `POST /inference` - Accepts image data and returns hand gesture predictions

### Infrastructure
- **Target Platform**: Raspberry Pi with Coral TPU
- **TPU**: Google Coral USB Accelerator for ML inference acceleration
- **Orchestration**: k3s (lightweight Kubernetes)
- **Cluster**: 3 Raspberry Pi nodes
- **Setup Tool**: k3sup (https://github.com/alexellis/k3sup)
- **Deployment**: Kubernetes manifests in `backend/k8s/backend.yaml`

## Key Components

### Inference Engine
- Supports TensorFlow models optimized for Coral TPU
- Model artifacts retrieved via API (requires `API_KEY` environment variable)
- Model name: "hand-rekog"
- Returns bounding box predictions for detected hands

### Dependencies
- Multiple requirements files for different environments:
  - `backend/requirements.txt` - General backend dependencies
  - `backend/requirements.backend.arm64.txt` - ARM64-specific backend dependencies
  - `coral/requirements.coral.arm64.txt` - Coral TPU dependencies for ARM64

## Development Workflow

### Local Development

**Frontend:**
```bash
streamlit run streamlit.py
```

**Backend (Mac/local):**
```bash
cd backend/
uvicorn main:app
```

### Kubernetes Deployment

**Prerequisites:**
- k3s cluster configured and running
- `kubectl` installed and configured

**Deploy:**
```bash
kubectl apply -f backend/k8s/backend.yaml
```

## Environment Variables

- `API_KEY` - Required for model artifact retrieval

## File Structure

```
hand-rekog-app/
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── backend.py           # Inference engine implementation
│   ├── requirements.txt     # Python dependencies
│   ├── requirements.backend.arm64.txt
│   └── k8s/
│       └── backend.yaml     # Kubernetes deployment manifest
├── coral/
│   └── requirements.coral.arm64.txt
├── streamlit.py             # Streamlit frontend application
└── README.md

```

## Technology Stack

- **ML Framework**: TensorFlow (with Coral TPU optimization)
- **Backend Framework**: FastAPI
- **Frontend Framework**: Streamlit
- **ASGI Server**: Uvicorn
- **Hardware Accelerator**: Google Coral TPU
- **Container Orchestration**: k3s (Kubernetes)
- **Target Hardware**: Raspberry Pi (ARM64)

## Important Notes

- The application is designed specifically for edge deployment on resource-constrained devices
- Coral TPU support requires specific TensorFlow Lite models (.tflite format)
- The inference backend expects base64-encoded image data in requests
- Models are fetched dynamically from a remote API rather than bundled with the application

## Future Considerations

- Frontend currently runs only locally; could be containerized and deployed to k8s
- Consider adding model caching to reduce API calls
- Monitor inference performance and TPU utilization
- Implement proper error handling and logging for production deployment
