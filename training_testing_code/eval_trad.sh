#!/bin/bash
#Runs all the hand-crafted methods and saves signals, then calculates stats for the still scenarios, just change -s argument to get stats for any scenarios set
python eval_get_signals_trad.py green
python eval_get_stats_30s.py -t output_signals/green/ -s still
python eval_get_signals_trad.py chrom
python eval_get_stats_30s.py -t output_signals/chrom/ -s still
python eval_get_signals_trad.py pos
python eval_get_stats_30s.py -t output_signals/pos/ -s still
python eval_get_signals_trad.py lgi
python eval_get_stats_30s.py -t output_signals/lgi/ -s still
python eval_get_signals_trad.py pca
python eval_get_stats_30s.py -t output_signals/pca/ -s still
python eval_get_signals_trad.py ica
python eval_get_stats_30s.py -t output_signals/ica/ -s still
