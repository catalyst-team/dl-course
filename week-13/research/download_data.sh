#!/usr/bin/env bash

DATA_PATH=$1

wget -q https://www.dropbox.com/s/te976klxsc84vqx/data_cat_dogs.zip
unzip -q data_cat_dogs.zip -d $DATA_PATH