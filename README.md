# hpc-primer

## 1. Setup a Dockerfile

### Install Docker

[https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/)

[https://docs.docker.com/engine/install/linux-postinstall/](https://docs.docker.com/engine/install/linux-postinstall/)

### Build Dockerfile and push to Docker Hub

Define a Dockerfile with the python environment that you'd like to use. This repository contains a sample Dockerfile. Build the Dockerfile locally using

    docker build -t <hub-user>/<repo-name>:<tag> .

Afterwards push the image to Docker Hub (make an account if necessary).

    docker push <hub-user>/<repo-name>:<tag>

## HPC

Login to HPC Cluster

    ssh <user>@gateway.hpc.tu-berlin.de

start singularity (a container platform similar to Docker)

    module load singularity/3.7.0

pull image from Dockerhub

    singularity pull docker://<hub-user>/<repo-name>:<tag>

Run the script:

    sbatch run.slurm