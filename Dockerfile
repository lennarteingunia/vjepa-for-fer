FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN mkdir jepaslt
RUN python -m pip install lightning
WORKDIR "/jepaslt"