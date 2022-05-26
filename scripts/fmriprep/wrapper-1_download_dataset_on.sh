#!/bin/bash

# INPUT ARGUMENTS:
# $1: dataset ID
# $2: dataset storage path
# $3 freesurfer license path (e.g., /home/athms/license.txt)
while [ $# -gt 0 ] ; do
  case $1 in
    -d | --dataset) DATASET="$2" ;;
    -p | --storage-path) STORAGE_DIR="$2" ;;
    -f | --freesurfer-license) SINGULARITYENV_FS_LICENSE="$2" ;;
  esac
  shift
done

# define other directories
export DATASET=$DATASET
export STUDY_DIR="$STORAGE_DIR/$DATASET"
export BIDS_DIR="${STUDY_DIR}/data"
export DERIVS_DIR="derivatives/fmriprep-20.2.3"

# download dataset
job_id1=$(sbatch --parsable scripts/fmriprep/download_dataset_on.sh)

# build singularity image
job_id2=$(sbatch --parsable scripts/fmriprep/build_fmriprep_image.sh)

# cleanup dataset dir
job_id3=$(sbatch --parsable --dependency=afterok:$job_id1 scripts/fmriprep/cleanup_bids_dir.sh)

