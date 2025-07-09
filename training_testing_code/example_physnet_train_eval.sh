#!/bin/bash
#Trains the Physnet method and saves signals, then calculates stats for the still scenarios, just change -s argument to get stats for any scenarios set
#To train in the AA scenario, -ts need to be set to all50 first and then all100. For passive scenarios stats are average over both, for active (I1-I6,M1-M10) only cross-frequency validated values are taken
python train_e2e.py -f 1 -m physnet -ts still
python eval_get_signals_e2e.py -l records/model/physnet__still__1/last.pth
python train_e2e.py -f 2 -m physnet -ts still
python eval_get_signals_e2e.py -l records/model/physnet__still__2/last.pth
python train_e2e.py -f 3 -m physnet -ts still
python eval_get_signals_e2e.py -l records/model/physnet__still__3/last.pth
python train_e2e.py -f 4 -m physnet -ts still
python eval_get_signals_e2e.py -l records/model/physnet__still__4/last.pth
python train_e2e.py -f 5 -m physnet -ts still
python eval_get_signals_e2e.py -l records/model/physnet__still__5/last.pth
python eval_get_stats_30s.py -t output_signals/physnet/ -s still
