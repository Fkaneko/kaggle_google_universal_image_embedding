FROM nvcr.io/nvidia/pytorch:22.02-py3

# Install in the default python3 environment

COPY ./requirements.txt /temp/
# COPY ./requirements_dev.txt /temp/
WORKDIR /temp/

# RUN pip install --quiet --no-cache-dir  -r requirements_dev.txt
RUN pip install --quiet --no-cache-dir  -r requirements.txt
# to get webp support https://github.com/ContinuumIO/anaconda-issues/issues/10737
RUN pip install --quiet --no-cache-dir  -U Pillow
#
# https://stackoverflow.com/questions/72706073/attributeerror-partially-initialized-module-cv2-has-no-attribute-gapi-wip-gs
RUN pip install --quiet --no-cache-dir opencv-python==4.5.5.64
# add nltk don
COPY ./scripts/download_nltk_database.py /temp/
RUN python download_nltk_database.py

RUN rm  /temp/*
