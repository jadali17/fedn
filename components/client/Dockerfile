FROM python:3.8.6
RUN pip install numpy==1.18.5 h5py==2.10.0 keras==2.4.3 pandas tensorflow==2.3.1 sklearn
RUN mkdir -p /app/client
RUN mkdir /app/certs
COPY fedn /app/fedn
RUN pip install -e /app/fedn
WORKDIR /app
