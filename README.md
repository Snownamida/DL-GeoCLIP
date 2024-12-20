# GeoCLIP
This is a CLIP Model to classify the photos by Spain Comunidad.

## Run with docker
```bash
docker run -d -p 23333:2333 -p 8000:8000 snownamida/geoclip:2.0  /root/start.sh
```
Then you can access the web interface by http://localhost:23333

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
python run_inference.py
```