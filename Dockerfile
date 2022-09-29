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

RUN chmod +x start.sh
CMD ["./start.sh"]