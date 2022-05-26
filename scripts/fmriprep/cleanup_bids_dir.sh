#!/bin/bash

# check if participants.tsv exists, if not write it
PARTICIPANTS_LIST="${STUDY_DIR}/data/participants.tsv"
if [[ ! -e "${PARTICIPANTS_LIST}" ]]; then
	printf "PARTICIPANTS\t\n" >> $PARTICIPANTS_LIST
	for subject in $STUDY_DIR/data/sub-*; do printf "$(basename $subject)\t\n" >> $PARTICIPANTS_LIST; done
fi

# check if any file with _events.json exists, if so move them out
find "${STUDY_DIR}/data/" -name '*_events.json' -exec mv {} "${STUDY_DIR}/" \;

# remove annex-uid file
if [[ -e "${STUDY_DIR}/data/annex-uuid" ]]; then
	rm "${STUDY_DIR}/data/annex-uuid"
fi