#!/bin/bash

aws s3 sync --no-sign-request "s3://openneuro.org/$DATASET" "$STUDY_DIR/data"