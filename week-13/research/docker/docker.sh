#!/usr/bin/env bash

WORK_DIR=$1

docker build . -t dl-course
docker run -v $WORK_DIR:/app -i -t dl-course /bin/bash