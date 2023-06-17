# SBN-evaluation-tool
This repository is specially designed for SBN parsing evaluation. This is a part of the work for the paper " Discourse Representation Structure Parsing for Chinese".

Since the evaluation metric of the original [SBN evaluation tool](https://github.com/WPoelman/ud-boxer) is too inflated, it is difficult to detect the difference between the pros and cons of different neural network models in the evaluation results.
In order to compress the evaluation results, we change the WordNets representation and Constants representation in the Penmant format and convert the Penman format to a coarser-grained format for SBN. 
<div align="center">
	<img src=".penman_format.png" alt="Editor" width="500">
</div>

To evaluate the quality of specific subtasks in DRS parsing, we imitate the fine-grained metrics for [AMR parsing task incremental](https://github.com/mdtux89/amr-evaluation), to DRS parsing. 
In order to make them compatible with DRS, we make some changes based on the data characteristics of DRS.  
Our fine-grained metrics consist of three categories in total: graph-level, node-level and edge-level. 
Each category includes more fine-grained evaluation metrics. All the metrics are proposed based on the semantic information types involved in DRS.


### Usage
```
cd 1.evaluation-tool-overall
sh run.sh en test /home/p-number/project/  ## $LANG,  $DATASET and $path for this project
```
### Details

- Our codes in ```sbnfile2eval.py``` are designed for English data and Chinese data evaluation, if other data needs to be evaluated, the file ```gold_output/path.out.$LANG.$DATA``` should be changed, which contains gold data and their paths. 

- In the ```./1.evaluation-tool-overall/results``` directory, you will find many files, ```$LANG.test.penman``` and ```$LANG.gold.penman``` are files generated by model and gold linearized SBN, respectively. 

- There are two Smatch scores will be calculated by ```sbnfile2eval.py```, "strict score" is the updated evaluation metric designed by us, which is used for fine-grained evaluation. "lenient score" is the original evaluation metric, designed by Wessel. You could find both formats of Penman in ```./1.evaluation-tool-overall/results``` directory.

- It should be noted that because some models' output may not be converted to Penman format due to format errors, in order to ensure the accuracy of fine-grained evaluation in ```2.evaluation-tool-detail```, the corresponding gold SBN is not converted to Penman format too. But it also means that the Smatch score may be different from the overall Smatch score calculated by ```1.evaluation-tool-overall``.

### Cite
If you use it, please cite the paper: Discourse Representation Structure Parsing for Chinese. Chunliu Wang, Xiao Zhang and Johan Bos. NATURAL LOGIC MEETS MACHINE LEARNING IV workshop. 2023, Nancy.

If you have any questions, please send an email to "springwillow.wang@gmail.com" or "chunliu.wang@rug.nl". 
