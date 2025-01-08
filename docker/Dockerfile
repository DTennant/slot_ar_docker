FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

COPY ../slot_ar /code/slot_ar

SHELL ["/bin/bash", "-c"]

RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0

RUN conda env create -n slot_ar -f /code/slot_ar/titok.yml && conda install -c conda-forge starship jupyterlab black git-lfs -y && git lfs install && git config --global credential.helper store && conda activate slot_ar && which pip && pip install -r req.txt && conda init bash

SHELL ["/opt/conda/bin/conda", "run", "-n", "slot_ar", "/bin/bash", "-c"]

# load imagenet
# 

WORKDIR /code/slot_ar
ENTRYPOINT ["bash", "pixart.sh"]



