#  Downloads the TensorFlow docker base image
FROM tensorflow/tensorflow:2.2.0rc2-gpu-py3-jupyter

# Install sagemaker-training toolkit to enable SageMaker Python SDK
RUN pip3 install sagemaker-training

# Copies training data inside the container
COPY training_data_dl.csv /opt/ml/code/training_data_dl.csv 

# Copies the training code inside the container
COPY ./scripts/train.py /opt/ml/code/train.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py
