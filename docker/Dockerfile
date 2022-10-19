# syntax=docker/dockerfile:1
FROM dockerhub.deepmagic.sk/stu/dev-base:latest

# Instalujeme inno packages
RUN apt-get update --fix-missing
RUN apt-get install apt-transport-https
RUN apt-get install ca-certificates
RUN update-ca-certificates

# User created in parent container
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

USER $USERNAME