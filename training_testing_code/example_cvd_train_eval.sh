#!/bin/bash
#Trains the CVD method and saves signals, then calculates stats for the still scenarios, just change -s argument to get stats for any scenarios set
#To train in the AA scenario, -ts need to be set to all50 first and then all100. For passive scenarios stats are average over both, for active only cross-frequency validated values are taken
python train_non-e2e.py -f 1 -m cvd -ts still
python eval_get_signals_non-e2e.py -l records/model/cvd__still__1/last.pth
python train_non-e2e.py -f 2 -m cvd -ts still
python eval_get_signals_non-e2e.py -l records/model/cvd__still__2/last.pth
python train_non-e2e.py -f 3 -m cvd -ts still
python eval_get_signals_non-e2e.py -l records/model/cvd__still__3/last.pth
python train_non-e2e.py -f 4 -m cvd -ts still
python eval_get_signals_non-e2e.py -l records/model/cvd__still__4/last.pth
python train_non-e2e.py -f 5 -m cvd -ts still
python eval_get_signals_non-e2e.py -l records/model/cvd__still__5/last.pth
python eval_get_stats_30s.py -t output_signals/cvd/ -s still
