FROM swarm.azurecr.io/swarm-dnn-nextgen:latest-cuda-122
WORKDIR /workspace

ENV FIFTYONE_API_URI="https://fiftyone-api.int.svc.bernard-gruppe.com"

RUN pip uninstall -y opencv-python
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "yolov11n.py"]