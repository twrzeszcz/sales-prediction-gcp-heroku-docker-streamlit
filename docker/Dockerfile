FROM python:3.7

RUN mkdir templates
RUN mkdir static

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt 

COPY test.json ./test.json
COPY cat_tuned.pkl ./cat_tuned.pkl
COPY prep_pipe_full.pkl ./prep_pipe_full.pkl
COPY utils.py ./utils.py

COPY index.html ./templates/index.html
COPY style.css ./static/style.css

COPY api.py ./api.py
