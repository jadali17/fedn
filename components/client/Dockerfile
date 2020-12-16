<<<<<<< HEAD
FROM python:3.8.6
RUN pip install numpy==1.18.5 h5py==2.10.0 keras==2.4.3 pandas tensorflow==2.3.1 sklearn
RUN mkdir -p /app/client
RUN mkdir /app/certs
=======
FROM scaleoutsystems/fedn-base:latest

COPY components/client/requirements.txt /app/requirements.txt
RUN mkdir -p /app && \
    mkdir -p /app/client && \
    mkdir -p /app/certs
>>>>>>> 893ddbe0710248038cca40f1608c28950c24ffe0
COPY fedn /app/fedn
RUN pip install -r /app/requirements.txt && \
    pip install -e /app/fedn

WORKDIR /app

