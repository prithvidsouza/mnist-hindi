# Base Image
FROM python:3.7

# Working Directory
WORKDIR /usr/src/app

# Dependencies
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install numpy
RUN pip install Pillow
RUN pip install pandas
RUN pip install tensorflow==1.14.0
