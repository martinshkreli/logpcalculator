FROM python:3.10.7-bullseye
 
RUN apt-get update && apt-get install -y \
    libopenbabel-dev \
    libopenbabel7 \
    openbabel \
    swig
 
WORKDIR /app
 
RUN ln -s /usr/include/openbabel3 /usr/local/include/openbabel3
 
RUN pip install openbabel swig
 
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app

CMD ["uvicorn", "server:app", "--host=0.0.0.0", "--port=$PORT"]