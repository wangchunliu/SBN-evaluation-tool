#!/bin/bash

# Evaluation script. Run as: ./evaluation.sh <parsed_data> <gold_data>

out=`python $3/SBN-evaluation-tool/2.evaluation-tool-detail/smatch/smatch.py --pr -f "$1" "$2"`
pr=`echo $out | cut -d' ' -f2`
rc=`echo $out | cut -d' ' -f4`
fs=`echo $out | cut -d' ' -f6`
echo 'Smatch -> P: '$pr', R: '$rc', F: '$fs

sed 's/:[A-Z][a-z][a-z]*/:role/g' "$1" > 1.tmp
sed 's/:[A-Z][a-z][a-z]*/:role/g' "$2" > 2.tmp
out=`python $3/SBN-evaluation-tool/2.evaluation-tool-detail/smatch/smatch.py --pr -f 1.tmp 2.tmp`
pr=`echo $out | cut -d' ' -f2`
rc=`echo $out | cut -d' ' -f4`
fs=`echo $out | cut -d' ' -f6`
echo 'No Roles -> P: '$pr', R: '$rc', F: '$fs
#
sed 's/:[A-Z]\{4,12\} /:relation /g' "$1" > 1.tmp
sed 's/:[A-Z]\{4,12\} /:relation /g' "$2" > 2.tmp
out=`python $3/SBN-evaluation-tool/2.evaluation-tool-detail/smatch/smatch.py --pr -f 1.tmp 2.tmp`
pr=`echo $out | cut -d' ' -f2`
rc=`echo $out | cut -d' ' -f4`
fs=`echo $out | cut -d' ' -f6`
echo 'No Discourse -> P: '$pr', R: '$rc', F: '$fs

sed 's/:[A-Z]\{3\} /:operator /g' "$1" > 1.tmp
sed 's/:[A-Z]\{3\} /:operator /g' "$2" > 2.tmp
out=`python $3/SBN-evaluation-tool/2.evaluation-tool-detail/smatch/smatch.py --pr -f 1.tmp 2.tmp`
pr=`echo $out | cut -d' ' -f2`
rc=`echo $out | cut -d' ' -f4`
fs=`echo $out | cut -d' ' -f6`
echo 'No Operators -> P: '$pr', R: '$rc', F: '$fs
#
cat "$1" | perl -ne 's/(.+)\.(n|v|a|r)\.(\d+)/\1/g; print;' > 1.tmp
cat "$2" | perl -ne 's/(.+)\.(n|v|a|r)\.(\d+)/\1/g; print;' > 2.tmp
out=`python $3/SBN-evaluation-tool/2.evaluation-tool-detail/smatch/smatch.py --pr -f 1.tmp 2.tmp`
pr=`echo $out | cut -d' ' -f2`
rc=`echo $out | cut -d' ' -f4`
fs=`echo $out | cut -d' ' -f6`
echo 'No Senses -> -> P: '$pr', R: '$rc', F: '$fs

cat "$1" | perl -ne 's/^#.*\n//g; print;' | tr '\t' ' ' | tr -s ' ' > 1.tmp
cat "$2" | perl -ne 's/^#.*\n//g; print;' | tr '\t' ' ' | tr -s ' ' > 2.tmp
python $3/SBN-evaluation-tool/2.evaluation-tool-detail/scores_nodes.py "1.tmp" "2.tmp"
python $3/SBN-evaluation-tool/2.evaluation-tool-detail/scores_triples.py "1.tmp" "2.tmp"

rm 1.tmp
rm 2.tmp
