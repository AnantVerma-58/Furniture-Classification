FROM python:3.12.8
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
ENTRYPOINT [ "streamlit", "run","app.py","--server.port=9999", "--server.address=0.0.0.0"]
