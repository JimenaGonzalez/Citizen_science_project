#!/bin/bash
source /cvmfs/des.opensciencegrid.org/eeups/startupcachejob31i.sh

TILES_PATH="/pnfs/des/persistent/sgonzal/vit_multiclass/Y6_tiles/"

input="all_tiles.txt"
while IFS= read -r line
do
  TILENAME=${line::-5}
  DETECTION_PATH="/pnfs/des/persistent/sgonzal/vit_multiclass/Y6_detections/"
  OUTPUT_PATH="$DETECTION_PATH""$TILENAME"".out"
  echo ${TILES_PATH}${line}
  if [[ -f "$OUTPUT_PATH" ]] && [[ -f "${TILES_PATH}${line}" ]]; then
    ifdh rm ${TILES_PATH}${line}
  fi
done < "$input"
