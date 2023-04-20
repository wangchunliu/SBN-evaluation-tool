#!/bin/bash
LANG=$1 ## en, zh
DATA=$2 ## dev, test
if [ -d "./results" ]; then
   rm ./results/$LANG.*.penman
fi
mkdir -p  ./results/

python ~/SBN-evaluation/1.evaluation-tool-overall/sbnfile2eval.py -gold  ~/SBN-evaluation/1.evaluation-tool-overall/gold_output/path.out.$DATA -test  ~/SBN-evaluation/1.evaluation-tool-overall/model_output/$LANG.sbn.output.$DATA -lang $LANG

sh ~/SBN-evaluation/2.evaluation-tool-detail/evaluation.sh ~/SBN-evaluation/1.evaluation-tool-overall/results/$LANG.test.penman ~/SBN-evaluation/1.evaluation-tool-overall/results/$LANG.gold.penman

