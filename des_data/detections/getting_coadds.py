#!/usr/bin/env python

import pandas as pd
import os
from os import path
from tqdm import tqdm

path = "/pnfs/des/persistent/sgonzal/vit_multiclass/Y6_detections/"
tilefile = '/data/des90.a/data/sgonzal/test_jobs/all_tiles.csv'
categories = ["_Positives.csv", "_Rings.csv", "_Companion.csv"]
category = categories[2]

dataframe = pd.read_csv(tilefile)
all_data = pd.DataFrame()

with tqdm(total=len(dataframe)) as pbar:
    for (index, row) in dataframe.iterrows():
        #if(index == 20): break
        tilename = row['tile'][:-5]
        if(os.path.exists(path + tilename + category ) == False):
            continue
        data_tmp = pd.read_csv(path + tilename + category)
        all_data = all_data.append(data_tmp, ignore_index=True)
        pbar.update(1)
print(len(all_data))
all_data.to_csv("coadd_ids" + category, index=False)
