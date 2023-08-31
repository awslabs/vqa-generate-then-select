# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random
import sys
from collections import Counter
import os


def process_answer_list(answers):
  new_list = []
  for answer in answers:
    new_ans = process_answer(answer)
    if new_ans not in new_list and new_ans:
      new_list.append(new_ans)
  return new_list


def process_answer(answer):
  answer = answer.replace('.', '').replace(',', '').lower()
  to_be_removed = {'a', 'an', 'the', 'to', ''}
  answer_list = []
  for an in answer.split(' '):
    if an not in to_be_removed:
      answer_list.append(an)
  return ' '.join(answer_list)


def get_stat(gen_answers_all, labels):
  highest_accus = []
  Ns = []
  for gen_answers in [[[g[0]] for g in gen_answers_all], [g[:3] for g in gen_answers_all], [g[:5] for g in gen_answers_all], gen_answers_all]:
    N = len(gen_answers)
    highest_accu = 0
    recall = 0
    for i, pred_answer in enumerate(gen_answers):
      highest = 0
      for pred_a in pred_answer:
        current = labels[i].get(process_answer(pred_a), 0.0)
        if current > highest: highest = current
      highest_accu += highest
      if highest > 0: recall += 1
    highest_accu = round(highest_accu / N * 100, 1)
    highest_accus.append(f'{highest_accu}')
    Ns.append(N)
  print(Ns, f'recall is {round(recall/ N*100, 1)}, top 1,3,5,all highest accus are {" & ".join(highest_accus)}')
  return highest_accus[-1]


def load_candidates(prediction):

  def get_cands(all_ans__):
    to_add = []
    to_add_all = []
    for ans in all_ans__.split(', or '):
      ans = ans.strip().replace('.', '')
      to_add_all.append(ans)
      if ans in to_add:
        continue
      else:
        to_add.append(ans)
    return to_add, to_add_all

  gen_answers_all = []
  gen_answers = []
  gen_answers_qc = []
  gen_answers_q = []
  for (k, v) in prediction.items():
    all_ans = v[0].replace('<pad>', '').replace('<extra_id_0>', '').split(', or or or, ')
    if len(all_ans) > 1:
      all_ans_qc = all_ans[0].split('===')[0].replace('=', '')
      all_ans_q = all_ans[1].split('===')[0].replace('=', '')
      all_ans_ = all_ans_qc + ', or ' + all_ans_q
      gen_answers_qc.append(get_cands(all_ans_qc)[0])
      gen_answers_q.append(get_cands(all_ans_q)[0])
    else:
      all_ans_ = all_ans[0].split('===')[0].replace('=', '')
    to_add_, to_add_all_ = get_cands(all_ans_)
    gen_answers.append(to_add_)
  return gen_answers, gen_answers_qc, gen_answers_q


def load_label(prediction):
  gold_answers = [v[2] for (k, v) in prediction.items()]
  labels = []
  for gold in gold_answers:
    label = {}
    counter = Counter(gold.split(';;'))
    for k, v in counter.items():
      label[k] = min(1., float(v) * 0.3)
    labels.append(dict(sorted(label.items(), key=lambda item: item[1], reverse=True)))
  return labels
