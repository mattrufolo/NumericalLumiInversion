#!/bin/bash
source activate mruf_env
cd /var/data/mrufolo/Inverting_luminosity/inv_gauss_tree_maker/000_noerrxy/003
python 000_simple_job.py > output.txt 2> error.txt
