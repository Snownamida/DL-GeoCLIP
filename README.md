# GeoCLIP
This is a CLIP Model to classify the photos by Spain Comunidad.

## Run with docker
```bash
docker run -d -p 2333:2333 -p 8000:8000 snownamida/geoclip:2.0  /root/start.sh
```
Then you can access the web interface by http://localhost:2333

## Manual Installation
Install lfs before pull the repository.
```bash
git lfs install
```
Install the requirements.
```bash
pip install -r requirements.txt
```
Then run the demo!
```bash
uvicorn run_inference:app --host 0.0.0.0 --port 8000 
python3 -m http.server 2333 --directory Project 
```
Then you can access the web interface by http://localhost:2333
