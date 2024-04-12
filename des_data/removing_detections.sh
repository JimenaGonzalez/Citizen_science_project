#!/bin/bash

detections_path="/pnfs/des/persistent/sgonzal/vit_multiclass/Y6_detections/"

for k in ${detections_path}*; do
    echo "$k"
    ifdh rm "$k"
done
