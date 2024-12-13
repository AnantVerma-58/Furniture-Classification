FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
ENTRYPOINT [ "streamlit", "run","app.py","--server.port=9999", "--server.address=0.0.0.0"]
