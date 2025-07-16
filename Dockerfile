FROM osrf/ros:humble-desktop

RUN apt update && apt install -y xvfb

WORKDIR /root/lydlr
