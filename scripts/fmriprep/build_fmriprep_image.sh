#!/bin/bash

# make sure dir for image exists
mkdir -p "$STUDY_DIR/images/"

# build image
singularity build "$STUDY_DIR/images/fmriprep-20.2.3.simg" docker://nipreps/fmriprep:20.2.3