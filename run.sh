#!/bin/bash

sudo docker run -i -t --rm -p 5000:5000 -v "$PWD/ml_server/data:/root/ml_server/data" --rm -i ml_server
