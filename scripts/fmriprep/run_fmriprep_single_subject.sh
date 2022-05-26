#!/bin/bash

# Prepare some writeable bind-mount points.
TEMPLATEFLOW_HOST_HOME=${HOME}/.cache/templateflow
FMRIPREP_HOST_CACHE=${HOME}/.cache/fmriprep
mkdir -p ${TEMPLATEFLOW_HOST_HOME}
mkdir -p ${FMRIPREP_HOST_CACHE}

# Prepare derivatives folder
mkdir -p "${BIDS_DIR}/${DERIVS_DIR}"
mkdir -p "${SCRATCH}/work/${DATASET}"

# Make sure FS_LICENSE is defined in the container.
export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer.txt

# Designate a templateflow bind-mount point
export SINGULARITYENV_TEMPLATEFLOW_HOME="/templateflow"
SINGULARITY_CMD="singularity run --cleanenv -B ${BIDS_DIR}:/data -B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} -B ${SCRATCH}/work/${DATASET}:/work $STUDY_DIR/images/fmriprep-20.2.3.simg"

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

# Compose the command line
cmd="${SINGULARITY_CMD} /data /data/${DERIVS_DIR} participant --participant-label $subject -w /work/ -vv --omp-nthreads 16 --nthreads 16 --mem_mb 60000 --output-spaces MNI152NLin2009cAsym:res-2 anat fsnative fsaverage5 --use-aroma --fs-subjects-dir /fsdir --fs-no-reconall --skip_bids_validation"

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode