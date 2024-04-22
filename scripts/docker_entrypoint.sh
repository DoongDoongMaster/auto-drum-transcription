#!/bin/sh

cd src/
python run_model_serving.py

exec "$@"