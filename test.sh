#!/bin/bash

python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/bce/train -n 5 -o output/zaretzki/FC1/bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/bce/train -n 10 -o output/zaretzki/FC1/bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/bce/train -n 15 -o output/zaretzki/FC1/bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/bce/train -n 20 -o output/zaretzki/FC1/bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/weight_bce/train -n 5 -o output/zaretzki/FC1/weight_bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/weight_bce/train -n 10 -o output/zaretzki/FC1/weight_bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/weight_bce/train -n 15 -o output/zaretzki/FC1/weight_bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/weight_bce/train -n 20 -o output/zaretzki/FC1/weight_bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/mcc_bce/train -n 5 -o output/zaretzki/FC1/mcc_bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/mcc_bce/train -n 10 -o output/zaretzki/FC1/mcc_bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/mcc_bce/train -n 15 -o output/zaretzki/FC1/mcc_bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC1/preprocessed/test -m output/zaretzki/FC1/mcc_bce/train -n 20 -o output/zaretzki/FC1/mcc_bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/bce/train -n 5 -o output/zaretzki/FC2/bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/bce/train -n 10 -o output/zaretzki/FC2/bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/bce/train -n 15 -o output/zaretzki/FC2/bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/bce/train -n 20 -o output/zaretzki/FC2/bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/weight_bce/train -n 5 -o output/zaretzki/FC2/weight_bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/weight_bce/train -n 10 -o output/zaretzki/FC2/weight_bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/weight_bce/train -n 15 -o output/zaretzki/FC2/weight_bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/weight_bce/train -n 20 -o output/zaretzki/FC2/weight_bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/mcc_bce/train -n 5 -o output/zaretzki/FC2/mcc_bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/mcc_bce/train -n 10 -o output/zaretzki/FC2/mcc_bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/mcc_bce/train -n 15 -o output/zaretzki/FC2/mcc_bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC2/preprocessed/test -m output/zaretzki/FC2/mcc_bce/train -n 20 -o output/zaretzki/FC2/mcc_bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/bce/train -n 5 -o output/zaretzki/FC3/bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/bce/train -n 10 -o output/zaretzki/FC3/bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/bce/train -n 15 -o output/zaretzki/FC3/bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/bce/train -n 20 -o output/zaretzki/FC3/bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/weight_bce/train -n 5 -o output/zaretzki/FC3/weight_bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/weight_bce/train -n 10 -o output/zaretzki/FC3/weight_bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/weight_bce/train -n 15 -o output/zaretzki/FC3/weight_bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/weight_bce/train -n 20 -o output/zaretzki/FC3/weight_bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/mcc_bce/train -n 5 -o output/zaretzki/FC3/mcc_bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/mcc_bce/train -n 10 -o output/zaretzki/FC3/mcc_bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/mcc_bce/train -n 15 -o output/zaretzki/FC3/mcc_bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC3/preprocessed/test -m output/zaretzki/FC3/mcc_bce/train -n 20 -o output/zaretzki/FC3/mcc_bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/bce/train -n 5 -o output/zaretzki/FC4/bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/bce/train -n 10 -o output/zaretzki/FC4/bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/bce/train -n 15 -o output/zaretzki/FC4/bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/bce/train -n 20 -o output/zaretzki/FC4/bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/weight_bce/train -n 5 -o output/zaretzki/FC4/weight_bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/weight_bce/train -n 10 -o output/zaretzki/FC4/weight_bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/weight_bce/train -n 15 -o output/zaretzki/FC4/weight_bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/weight_bce/train -n 20 -o output/zaretzki/FC4/weight_bce/test/20 -v INFO

python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/mcc_bce/train -n 5 -o output/zaretzki/FC4/mcc_bce/test/5 -v INFO
python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/mcc_bce/train -n 10 -o output/zaretzki/FC4/mcc_bce/test/10 -v INFO
python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/mcc_bce/train -n 15 -o output/zaretzki/FC4/mcc_bce/test/15 -v INFO
python scripts/test.py -d data/zaretzki/FC4/preprocessed/test -m output/zaretzki/FC4/mcc_bce/train -n 20 -o output/zaretzki/FC4/mcc_bce/test/20 -v INFO
