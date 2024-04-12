#!/bin/bash

OLDHOME=$HOME
export HOME=$PWD
/bin/hostname
source /cvmfs/des.opensciencegrid.org/eeups/startupcachejob31i.sh
IFDH_CP_MAXRETRIES=1
ifdh cp -D /pnfs/des/persistent/sgonzal/vit_multiclass/tiles_n.txt .

NUM=$(($PROCESS+1))
echo "$NUM"
TILE=$(awk '(NR=='$NUM') {print $0}' tiles_n.txt)
TILE=${TILE::-5}

ifdh cp -D /pnfs/des/persistent/sgonzal/vit_multiclass/model.pt .
ifdh cp -D /pnfs/des/persistent/sgonzal/vit_multiclass/pytorch_env.tar.gz .
ifdh cp -D /pnfs/des/persistent/sgonzal/vit_multiclass/Search_per_tile_multi_final.py .
ifdh cp -D /pnfs/des/persistent/sgonzal/vit_multiclass/Y6_tiles/${TILE}.fits .
ifdh cp -D /pnfs/des/persistent/sgonzal/vit_multiclass/jx_vit_base_p16_224-80ecf9dd.pth .

mkdir pytorch_env
tar -xzf pytorch_env.tar.gz -C pytorch_env
. pytorch_env/bin/activate

python Search_per_tile_multi_final.py ${TILE}.fits > ${TILE}.out
ifdh cp -D ${TILE}.out /pnfs/des/persistent/sgonzal/vit_multiclass/Y6_detections/

categories=("_Positives.csv" "_Rings.csv" "_Companion.csv")

for category in ${categories[@]}; do
  FILE=${TILE}${category}
  if [ -f "$FILE" ]; then
    echo "$FILE exists."
    ifdh cp -D ${FILE} /pnfs/des/persistent/sgonzal/vit_multiclass/Y6_detections/
  fi
done
