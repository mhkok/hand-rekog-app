# hand-rekog-app
Webbased app to rekognise hand gestures through your camera, running on a Pi &amp; Coral TPU

# architecture
The application consists of a frontend and a backend api. The frontend is currently based on `streamlit` and simply is connecting to the backend using an `/inference` endpoint. The inference backend was build with the intention to run on a Coral TPU (https://coral.ai/docs/accelerator/get-started) enabled Raspberry Pi. The TPU enabled Pi is part of a Kubernetes cluster (k3s) consisting in total of 3 Pi's that make up the cluster. Each Pi is a Kubernetes node. The k3s cluster has been set up using k3sup (see here: https://github.com/alexellis/k3sup) 

# how do I run the app?
Ensure to install the `requirements` in a pyenv.

## frontend
Currently you can only run the frontend locally from your machine. The frontend you can run by executing `streamlit run streamlit.py` from the root of the repo.

## backend
The backend you can run either on Mac or k8s. 

To run locally run `uvicorn main:app` in the `/backend/`

To run on k8s run `kubectl apply -f backend.yaml` (assuming you have k8s cluster set up and `kubectl` properly installed)

