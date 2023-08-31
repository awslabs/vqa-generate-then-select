# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json, pickle
import random
from tqdm import tqdm
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


def get_kat_input(prediction, gpt_name, reason_path, train=0):
  gen_answers = load_candidates(prediction)[0]
  # get Kat stat
  add_reason = os.path.exists(reason_path)
  if train:
    gpt3_path = f'../KAT/okvqa/train2014/gpt3_okvqa_train2014_answers.pkl'
    output_file1 = f'../KAT/okvqa/train2014/gpt3{gpt_name}_okvqa_train2014_answers.pkl'
    output_file3 = f'../KAT/okvqa/train2014/{gpt_name}_okvqa_train2014_answers.pkl'
    output_file_reason1 = f'../KAT/okvqa/train2014/gpt3{gpt_name}r_okvqa_train2014_answers.pkl'
    output_file_reason3 = f'../KAT/okvqa/train2014/{gpt_name}r_okvqa_train2014_answers.pkl'
  else:
    gpt3_path = f'../KAT/okvqa/val2014/gpt3_okvqa_val2014_answers.pkl'
    output_file1 = f'../KAT/okvqa/val2014/gpt3{gpt_name}_okvqa_val2014_answers.pkl'
    output_file3 = f'../KAT/okvqa/val2014/{gpt_name}_okvqa_val2014_answers.pkl'
    output_file_reason1 = f'../KAT/okvqa/val2014/gpt3{gpt_name}r_okvqa_val2014_answers.pkl'
    output_file_reason3 = f'../KAT/okvqa/val2014/{gpt_name}r_okvqa_val2014_answers.pkl'
  gpt_answers = pickle.load(open(gpt3_path, 'rb'))
  if add_reason:
    reasonings = json.load(open(reason_path, "r"))
    reason = {i: d.split('Context:')[0] for (i, d) in enumerate(reasonings)}
    print(f'reason random example', reason[2222])
  plm_data = {}
  plm_data3 = {}
  if add_reason:
    plmr_data = {}
    plmr_data3 = {}
  for i, candidates in tqdm(enumerate(gen_answers)):
    key = list(gpt_answers.keys())[i]
    v = gpt_answers[key][0]
    cands = ', or '.join([v[0]] + process_answer_list(candidates))
    cands3 = ', or '.join(process_answer_list(candidates))
    plm_data[key] = [(cands, v[1])]
    plm_data3[key] = [(cands3, '')]
    if add_reason:
      plmr_data[key] = [(cands, v[1] + ' ' + reason.get(i, ''))]
      plmr_data3[key] = [(cands3, reason.get(i, ''))]
  pickle.dump(plm_data, open(output_file1, 'wb'))
  pickle.dump(plm_data3, open(output_file3, 'wb'))
  if add_reason:
    pickle.dump(plmr_data, open(output_file_reason1, 'wb'))
    pickle.dump(plmr_data3, open(output_file_reason3, 'wb'))


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
    # all_ans = v[0].replace('<pad>', '').replace('<extra_id_0>', '').split('===')[0].replace('=', '').split(', or or or, ')

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
    # gen_answers_all.append(to_add_all_)
  return gen_answers, gen_answers_qc, gen_answers_q  #, gen_answers_all


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


def get_unifiedQA_input(prediction, model_name, reason_path, train=False):
  add_reason = os.path.exists(reason_path)
  reason = {}
  if add_reason:
    reasonings = json.load(open(reason_path, "r"))
    reason = {i: d.split('Context:')[0] for (i, d) in enumerate(reasonings)}
    print(f'reason random example', reason[2222])
  input_reason = []
  gen_answers = load_candidates(prediction)[0]
  labels = load_label(prediction)

  choices = ['(A)', '(B)', '(C)', '(D)', '(E)']
  input = []
  if train:
    coco_caption = json.load(open('coco_annotations/captions_train2014.json', 'r'))['annotations']
    train_caption_dict = {}
    for sample in coco_caption:
      if sample['image_id'] not in train_caption_dict:
        train_caption_dict[sample['image_id']] = [sample['caption']]
      else:
        train_caption_dict[sample['image_id']].append(sample['caption'])
  for i, (k, v) in enumerate(prediction.items()):
    img_key = int(k.split('<->')[0])
    cands = gen_answers[i][:5]
    prompt = v[1]
    # gold_answer = v[2]
    gold_answer = list(labels[i].keys())[0]
    if train:
      caption = random.choice(train_caption_dict[img_key])
      cands = cands[:3]
      cands = list(set(cands + list(labels[i].keys())[:2]))
    else:
      context = prompt.split('\n===\nContext: ')[-1].split('\n===\nQ: ')[0]
      try:
        caption, tags = context.split('. ')
      except Exception as e:
        print(e)
        caption, tags = context, ''
    question = prompt.split('===\nQ: ')[-1].split('\nA:')[0]
    input_choices = ''
    random.shuffle(cands)
    for j in range(len(cands)):
      input_choices += f' {choices[j]} {cands[j]}'

    input.append([f"{question} \\n{input_choices} \\n {caption}", gold_answer])
    input_reason.append([f"{question} \\n{input_choices} \\n {caption} \\n {reason.get(i, '')}", gold_answer])
  output_path = f'../KAT/unifiedQA/train2014/{model_name}_okvqa.json' \
      if train else f'../KAT/unifiedQA/val2014/{model_name}_okvqa.json'
  json.dump(input, open(output_path, 'w'))
  if add_reason:
    output_pathr = f'../KAT/unifiedQA/train2014/{model_name}r_okvqa.json' \
        if train else f'../KAT/unifiedQA/val2014/{model_name}r_okvqa.json'
    json.dump(input_reason, open(output_pathr, 'w'))
  return input


if __name__ == '__main__':
  output_folder = './output'
  file_name = 'PICa_codex_vinvl_tag-1_n16_repeat1_imagequestion_0.001_multi_recall_78.73563218390805'
  # file_name = 'train_PICa_codex_vinvl_tag-1_n16_repeat1_imagequestion_0.001_multi_recall_69.74136974136974'
  output_path = f'{output_folder}/prompt_answer/{file_name}.json'

  reason_path = f'output/candidate_reasoning/{file_name.split("_multi")[0]+"_reason"}.json'
  prediction = json.load(open(output_path, "r"))
  if 'PICa' in file_name:
    model_name = file_name.split('PICa_')[1].split('_')[0]
  else:
    model_name = file_name.split('_')[0]
  gen_answers, gen_answers_qa, gen_answers_q = load_candidates(prediction)
  labels = load_label(prediction)

  get_kat_input(prediction, model_name, reason_path, train=bool('train' in file_name))
  data = get_unifiedQA_input(prediction, model_name, reason_path, train=bool('train' in file_name))
