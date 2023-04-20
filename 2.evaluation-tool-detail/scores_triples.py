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

c2c_pred = [] # concept role concept
c2c_gold = []
c2n_pred = [] # concept Name "Mary"
c2n_gold = []
b2c_pred = [] # Box Member concept
b2c_gold = []
c2o_pred = [] # concept Operator "now"
c2o_gold = []
b2b_pred = [] # Box1 NEGATION box2
b2b_gold = []

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

    c2c_pred.append(c2c(dict_pred, triples_pred)) # concept2concpet
    c2c_gold.append(c2c(dict_gold, triples_gold))

    c2n_pred.append(c2n(dict_pred, triples_pred))  # concept2name
    c2n_gold.append(c2n(dict_gold, triples_gold))

    b2c_pred.append(b2c(dict_pred, triples_pred))  # box2concept
    b2c_gold.append(b2c(dict_gold, triples_gold))

    c2o_pred.append(c2o(dict_pred, triples_pred)) # concept2constant
    c2o_gold.append(c2o(dict_gold, triples_gold))

    b2b_pred.append(b2b(dict_pred, triples_pred)) # box2box
    b2b_gold.append(b2b(dict_gold, triples_gold))


pr, rc, f = smatch.main(c2c_pred, c2c_gold, True)  # kill.v.01 Agent female.n.01
print('Roles_triple -> P:', "{0:.3f}".format(float(pr)), ', R:', "{0:.3f}".format(float(rc)), ', F:', "{0:.3f}".format(float(f)))

pr, rc, f = smatch.main(c2n_pred, c2n_gold, True) ##  female.n.01 Name "Mary"
print('Names_triple -> P:', "{0:.3f}".format(float(pr)), ', R:', "{0:.3f}".format(float(rc)), ', F:', "{0:.3f}".format(float(f)))

pr, rc, f = smatch.main(b2c_pred, b2c_gold, True) ## box1 member female.n.01
print('Members_triple -> P:', "{0:.3f}".format(float(pr)), ', R:', "{0:.3f}".format(float(rc)), ', F:', "{0:.3f}".format(float(f)))

pr, rc, f = smatch.main(c2o_pred, c2o_gold, True) ##  time.n.08 EQU now
print('Operators_triple -> P:', "{0:.3f}".format(float(pr)), ', R:', "{0:.3f}".format(float(rc)), ', F:', "{0:.3f}".format(float(f)))

pr, rc, f = smatch.main(b2b_pred, b2b_gold, True) ## box1 NEGATION box2
print('Discourses_triple -> P:', "{0:.3f}".format(float(pr)), ', R:', "{0:.3f}".format(float(rc)), ', F:', "{0:.3f}".format(float(f)))
