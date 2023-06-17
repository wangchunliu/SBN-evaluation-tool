#!/usr/bin/env python
#coding=utf-8

'''
Computes AMR scores for concept identification, named entity recognition, wikification,
negation detection, reentrancy detection and SRL.

@author: Marco Damonte (m.damonte@sms.ed.ac.uk)
@since: 03-10-16
'''

import sys
import smatch.amr as amr
import smatch.smatch_fromlists as smatch
from collections import defaultdict
from utils import *

pred = open(sys.argv[1]).read().strip().split("\n\n")
gold = open(sys.argv[2]).read().strip().split("\n\n")

inters = defaultdict(int)
golds = defaultdict(int)
preds = defaultdict(int)

for amr_pred, amr_gold in zip(pred, gold):
    amr_pred = amr.AMR.parse_AMR_line(amr_pred.replace("\n",""))
    dict_pred = var2concept(amr_pred)
    triples_pred = []
    for t in amr_pred.get_triples()[1] + amr_pred.get_triples()[2]:
        if t[0].endswith('-of'):
            triples_pred.append((t[0][:-3], t[2], t[1]))
        else:
            triples_pred.append((t[0], t[1], t[2]))

    amr_gold = amr.AMR.parse_AMR_line(amr_gold.replace("\n",""))
    dict_gold = var2concept(amr_gold)
    triples_gold = []
    for t in amr_gold.get_triples()[1] + amr_gold.get_triples()[2]:
        if t[0].endswith('-of'):
            triples_gold.append((t[0][:-3], t[2], t[1]))
        else:
            triples_gold.append((t[0], t[1], t[2]))
    list_pred = disambig(namedent(dict_pred, triples_pred))
    list_gold = disambig(namedent(dict_gold, triples_gold))
    inters["Names"] += len(list(set(list_pred) & set(list_gold)))
    preds["Names"] += len(set(list_pred))
    golds["Names"] += len(set(list_gold))

    list_pred = disambig(negations(dict_pred, triples_pred))
    list_gold = disambig(negations(dict_gold, triples_gold))
    inters["Negation"] += len(list(set(list_pred) & set(list_gold)))
    preds["Negation"] += len(set(list_pred))
    golds["Negation"] += len(set(list_gold))

    list_pred = disambig(discources(dict_pred, triples_pred))
    list_gold = disambig(discources(dict_gold, triples_gold))
    inters["Discourse"] += len(list(set(list_pred) & set(list_gold)))
    preds["Discourse"] += len(set(list_pred))
    golds["Discourse"] += len(set(list_gold))

    list_pred = disambig(roles(triples_pred))
    list_gold = disambig(roles(triples_gold))
    inters["Roles"] += len(list(set(list_pred) & set(list_gold)))
    preds["Roles"] += len(set(list_pred))
    golds["Roles"] += len(set(list_gold))

    list_pred = disambig(members(triples_pred))
    list_gold = disambig(members(triples_gold))
    inters["Members"] += len(list(set(list_pred) & set(list_gold)))
    preds["Members"] += len(set(list_pred))
    golds["Members"] += len(set(list_gold))

    list_pred = disambig(concepts(dict_pred))
    list_gold = disambig(concepts(dict_gold))
    inters["Concepts"] += len(list(set(list_pred) & set(list_gold)))
    preds["Concepts"] += len(set(list_pred))
    golds["Concepts"] += len(set(list_gold))

    list_pred = disambig(con_noun(dict_pred))
    list_gold = disambig(con_noun(dict_gold))
    inters["Con_noun"] += len(list(set(list_pred) & set(list_gold)))
    preds["Con_noun"] += len(set(list_pred))
    golds["Con_noun"] += len(set(list_gold))

    list_pred = disambig(con_adj(dict_pred))
    list_gold = disambig(con_adj(dict_gold))
    inters["Con_adj"] += len(list(set(list_pred) & set(list_gold)))
    preds["Con_adj"] += len(set(list_pred))
    golds["Con_adj"] += len(set(list_gold))

    list_pred = disambig(con_adv(dict_pred))
    list_gold = disambig(con_adv(dict_gold))
    inters["Con_adv"] += len(list(set(list_pred) & set(list_gold)))
    preds["Con_adv"] += len(set(list_pred))
    golds["Con_adv"] += len(set(list_gold))

    list_pred = disambig(con_verb(dict_pred))
    list_gold = disambig(con_verb(dict_gold))
    inters["Con_verb"] += len(list(set(list_pred) & set(list_gold)))
    preds["Con_verb"] += len(set(list_pred))
    golds["Con_verb"] += len(set(list_gold))


for score in preds:
    if preds[score] > 0:
        pr = inters[score]/float(preds[score])
    else:
        pr = 0
    if golds[score] > 0:
        rc = inters[score]/float(golds[score])
    else:
        rc = 0
    if pr + rc > 0:
        f = 2*(pr*rc)/(pr+rc)
        print(score, '-> P:', "{0:.3f}".format(pr), ', R:', "{0:.3f}".format(rc), ', F:', "{0:.3f}".format(f))
    else:
        print(score, '-> P:', "{0:.3f}".format(pr), ', R:', "{0:.3f}".format(rc), ', F: 0.00')
