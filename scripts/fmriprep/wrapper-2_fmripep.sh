#!/bin/bash

# INPUT ARGUMENTS:
# d: OpenNeuro dataset ID
# p: dataset storage path
# w: working directory
# f: freesurfer license path
while [ $# -gt 0 ] ; do
  case $1 in
    -d | --dataset) DATASET="$2" ;;
    -p | --storage-path) STORAGE_DIR="$2" ;;
    -w | --work-path) WORK_DIR="$2" ;;
    -f | --freesurfer-license) SINGULARITYENV_FS_LICENSE="$2" ;;
  esac
  shift
done

mkdir -p "$WORK_DIR/$DATASET"
rsync -av "$STORAGE_DIR/$DATASET" $WORK_DIR

# define other directories
export DATASET=$DATASET
export STUDY_DIR="$WORK_DIR/$DATASET"
export BIDS_DIR="${STUDY_DIR}/data"
export DERIVS_DIR="derivatives/fmriprep-20.2.3"

# run fmriprep for each subject w/ singularity
sbatch --array=1-$(( $( wc -l ${BIDS_DIR}/participants.tsv | cut -f1 -d' ' ) - 1 )) scripts/fmriprep/run_fmriprep_single_subject.sh