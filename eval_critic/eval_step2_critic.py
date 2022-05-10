###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# NOT WORKS FOR good sent == bad sent

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
import re
import os
import sys
import json
import numpy as np
import editdistance
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, '.')
from utils.text_utils import detokenize_sent
from critic.step2_critic import run_bert_model, critic, init_model


def load_data():
    data_path = 'eval_critic/eval_data.jsonl'
    good_sents, bad_sents = [], []
    for line in open(data_path):
        obj = json.loads(line)
        good_sents.append(obj['good'])
        bad_sents.append(obj['bad'])
    return good_sents, bad_sents


good_sents, bad_sents = load_data()


def get_logps(bert_model, bert_tokenizer, sents):
    final = []
    _e = 0
    for start in tqdm(range(0, len(sents), 1)):
        sents_sub = sents[start: start + 1]
        sents_sub_detok = [detokenize_sent(sent) for sent in sents_sub]
        logps = run_bert_model(bert_model, bert_tokenizer, sents_sub_detok)
        if logps is None:
            _e += 1
            print(_e)
            continue
        for i in range(len(sents_sub)):
            final.append(
                {'sent': sents_sub[i], 'sent_detok': sents_sub_detok[i],
                 'logp': float(logps[i])})
    return final


def evaluate_logp(model, tokenizer):
    """
    Check whether log p(bad_sent) < log p(good_sent)
    """
    good_logps = get_logps(model, tokenizer, good_sents)
    bad_logps = get_logps(model, tokenizer, bad_sents)
    accs = []
    for good, bad in zip(good_logps, bad_logps):
        accs.append(int(bad['logp'] < good['logp']))
    avg_acc = float(sum(accs)) / len(accs)
    print(
        f'log p(bad) < log p(good)? {sum(accs)} / {len(accs)} = {avg_acc:.3f}')
    return good_logps, bad_logps


def compute_metrics(good_accs, bad_accs):
    try:
        goodP = float(sum(good_accs)) / (
                len(bad_accs) - sum(bad_accs) + sum(good_accs))
    except ZeroDivisionError:
        goodP = 0

    try:
        goodR = float(sum(good_accs)) / len(good_accs)
    except ZeroDivisionError:
        goodR = 0

    try:
        goodF05 = (1 + 0.5 ** 2) * float(goodP * goodR) / (
                (0.5 ** 2 * goodP) + goodR)
    except ZeroDivisionError:
        goodF05 = 0

    try:
        badP = float(sum(bad_accs)) / (
                len(good_accs) - sum(good_accs) + sum(bad_accs))
    except ZeroDivisionError:
        badP = 0

    try:
        badR = float(sum(bad_accs)) / len(bad_accs)
    except ZeroDivisionError:
        badR = 0

    try:
        badF05 = (1 + 0.5 ** 2) * float(badP * badR) / (
                (0.5 ** 2 * badP) + badR)
    except ZeroDivisionError:
        badF05 = 0

    print(
        f'  Good precision = {sum(good_accs)} / {(len(bad_accs) - sum(bad_accs) + sum(good_accs))} = {goodP:.3f}')
    print(
        f'  Good recall    = {sum(good_accs)} / {len(good_accs)} = {goodR:.3f}')
    print(f'  Good F0.5      = {goodF05:.3f}')
    print(
        f'  Bad precision  = {sum(bad_accs)} / {(len(good_accs) - sum(good_accs) + sum(bad_accs))} = {badP:.3f}')
    print(f'  Bad recall     = {sum(bad_accs)} / {len(bad_accs)} = {badR:.3f}')
    print(f'  Bad F0.5       = {badF05:.3f}')
    return {'goodP': goodP, 'goodR': goodR, 'goodF05': goodF05, 'badP': badP,
            'badR': badR, 'badF05': badF05}


def evaluate_LM_Critic(gpt2_model, gpt2_tokenizer, bert_model, bert_tokenizer):
    good_accs, bad_accs = [], []
    for obj in tqdm(good_logps):
        res = critic(gpt2_model, gpt2_tokenizer, bert_model, bert_tokenizer,
                     obj['sent_detok'], verbose=0, seed=1,
                     n_samples=100,
                     word_level_mode='refine')
        try:
            pred = int(res[0])
        except TypeError as err:
            good_accs.append(False)
            continue
        good_accs.append(pred == 1)
    for obj in tqdm(bad_logps):
        res = critic(gpt2_model, gpt2_tokenizer, bert_model, bert_tokenizer,
                     obj['sent_detok'], verbose=0, seed=1,
                     n_samples=100,
                     word_level_mode='refine')
        try:
            pred = int(res[0])
        except TypeError as err:
            bad_accs.append(False)
            continue
        bad_accs.append(pred == 0)
    print('\nLM-Critic:')
    stats = compute_metrics(good_accs, bad_accs)
    # json.dump(stats, open('lm_critic.stats.json', 'w'), indent=2)

    return stats


model_name_list = [
    # F0.5 Good, F0.5 Bad
    # gpt2: 0.696, 0.711
    'bert-base-uncased',  # 0.523, 0.584
    'roberta-base',  # 0.588, 0.610
    'emilyalsentzer/Bio_ClinicalBERT',  # 0.321, 0.566
    'dmis-lab/biobert-base-cased-v1.2',  # 0.452, 0.578
]

for model_name in model_name_list:
    gpt2_model, gpt2_tokenizer, bert_model, bert_tokenizer = init_model(
        model_name)

    good_logps, bad_logps = evaluate_logp(bert_model, bert_tokenizer)

    stats = evaluate_LM_Critic(gpt2_model, gpt2_tokenizer, bert_model,
                               bert_tokenizer)

    model_name_base = model_name.split('/')[-1]
    with open(f'eval_critic/step2_{model_name_base}_output.jsonl', 'w') as f:
        json.dump(stats, f, indent=2)

# log p(bad) < log p(good)? 555 / 586 = 0.947


# LM-Critic: (there is variance due to the randomness of sampling, some variation in GPT2 return score)
#   Good precision = 446 / 654 = 0.682
#   Good recall    = 446 / 586 = 0.761
#   Good F0.5      = 0.696
#   Bad precision  = 378 / 518 = 0.730
#   Bad recall     = 378 / 586 = 0.645
#   Bad F0.5       = 0.711
