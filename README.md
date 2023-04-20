# SBN-evaluation-tool
This repository is specially designed for SBN parsing.

Since the evaluation metrics of the original [SBN evaluation tool](https://github.com/WPoelman/ud-boxer) is too inflated, it is difficult to detect the difference between the pros and cons of different neural network models in the evaluation results.
In order to compress the evaluation results, we change the WordNets representation and Constants representation in the Penmant format, and convert the Penman format of SBN to a coarser-grained Penman format.

To evaluate the quality of specific subtasks in DRS parsing, we imitate the fine-grained metrics for [AMR parsing task incremental](https://github.com/mdtux89/amr-evaluation), to DRS parsing. 
In order to make them compatible for DRS, we make some changes based on the data characteristics of DRS.  
Our fine-grained metrics consist of three categories in total: graph-level, node-level and edge-level. 
Each category includes more fine-grained evaluation metrics. All the metrics are proposed based on the semantic information types involved in DRS.


### Usage
```
cd 1.evaluation-tool-overall
sh run.sh en dev  ## $LANG and $ DATA
```
### Details

Our codes in ```sbnfile2eval.py``` are designed for English data and Chinese data evaluation, if you want to evaluation other data, you need change the file in ```gold_output/path.out.$DATA```, which contains gold data and their paths. 

You need to pay special attention to lines 191-196 in the ```sbnfile2eval.py```, and rewrite them according to your own reference data.


