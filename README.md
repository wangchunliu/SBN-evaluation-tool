# SBN-evaluation-tool
This repository is specially designed for SBN parsing.

Since the evaluation metric of the original [SBN evaluation tool](https://github.com/WPoelman/ud-boxer) is too inflated, it is difficult to detect the difference between the pros and cons of different neural network models in the evaluation results.
In order to compress the evaluation results, we change the WordNets representation and Constants representation in the Penmant format, and convert the Penman format of SBN to a coarser-grained Penman format.

To evaluate the quality of specific subtasks in DRS parsing, we imitate the fine-grained metrics for [AMR parsing task incremental](https://github.com/mdtux89/amr-evaluation), to DRS parsing. 
In order to make them compatible with DRS, we make some changes based on the data characteristics of DRS.  
Our fine-grained metrics consist of three categories in total: graph-level, node-level and edge-level. 
Each category includes more fine-grained evaluation metrics. All the metrics are proposed based on the semantic information types involved in DRS.


### Usage
```
cd 1.evaluation-tool-overall
sh run.sh en dev  ## $LANG and $DATA
```
### Details

- Our codes in ```sbnfile2eval.py``` are designed for English data and Chinese data evaluation, if other data needs to be evaluated, the file ```gold_output/path.out.$DATA``` should be changed, which contains gold data and their paths. 

- Need to pay special attention to lines 191-196 in the ```sbnfile2eval.py```, and rewrite them according to reference data.

- In ```./1.evaluation-tool-overall/results``` directory, you will find many files, ```$LANG.test.penman``` and ```$LANG.gold.penman``` are files generated by gold linearized SBN and model generated linearized SBN. 

- It should be noted that because some models' output may not be converted to Penman format due to format errors, in order to ensure the accuracy of fine-grained evaluation in ```2.evaluation-tool-detail```, the corresponding gold SBN is not converted to Penman format too. But it also means that the two Smatch scores may not match.


