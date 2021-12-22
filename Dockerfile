FROM python:3.8-slim

COPY ml_server/src/requirements.txt /root/ml_server/src/requirements.txt

RUN chown -R root:root /root/ml_server

WORKDIR /root/ml_server/src
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ml_server/src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP run_server.py

RUN chmod +x run_server.py
CMD ["python3", "run_server.py"]
