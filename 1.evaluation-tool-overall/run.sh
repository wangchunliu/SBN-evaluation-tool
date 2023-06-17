#!/bin/bash
LANG=$1 ## en, zh, trans
DATA=$2 ## dev, test
PATH=$3 ## path for whole project
if [ "$#" -lt 3 ]; then
  echo "./run.sh <lang_type> <data_type> <path>"
  exit 2
fi

if [ -d "./results" ]; then
   rm ./results/$LANG.*.penman
fi
mkdir -p  ./results/

## Example command, you can change the file name or path
python $PATH/SBN-evaluation-tool/1.evaluation-tool-overall/sbnfile2eval.py -gold  $PATH/SBN-evaluation-tool/1.evaluation-tool-overall/gold_output/path.gold.$LANG.$DATA -test  $PATH/SBN-evaluation-tool/1.evaluation-tool-overall/model_output/$LANG.sbn.output.$DATA -lang $LANG

sh $PATH/SBN-evaluation-tool/2.evaluation-tool-detail/evaluation.sh $PATH/SBN-evaluation-tool/1.evaluation-tool-overall/results/$LANG.test.penman $PATH/SBN-evaluation-tool/1.evaluation-tool-overall/results/$LANG.gold.penman $PATH

