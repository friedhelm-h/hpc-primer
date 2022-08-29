# syntax=docker/dockerfile:1
FROM pytorch/pytorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir matplotlib \
    pip install tensorboard \
    pip install wandb

