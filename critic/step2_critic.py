import sys
import torch
import random
import hashlib
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    AutoModelForMaskedLM, RobertaForCausalLM, BertLMHeadModel, \
    BartForConditionalGeneration

# sys.path.insert(0, '.')
from critic.perturbations import (
    get_local_neighbors_char_level, get_local_neighbors_word_level
)
from utils.spacy_tokenizer import spacy_tokenize_gec

MAX_LENGTH = 66


def init_model(
        model_name='bert-base-uncased',
        cuda: bool = True
):
    # gpt2
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.eval()

    # bert
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModelForMaskedLM.from_pretrained(model_name)
    bert_tokenizer.bos_token = bert_tokenizer.pad_token
    bert_model.eval()

    if cuda:
        gpt2_model.cuda()
        bert_model.cuda()

    print(f'Loaded gpt2 and {model_name}')

    return gpt2_model, gpt2_tokenizer, bert_model, bert_tokenizer


def compute_loss(
        model,
        input_ids,
        attention_mask,
        labels
):
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        lm_logits = outputs[1]  # [bsize, seqlen, vocab]
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(bsize, seqlen - 1)
            loss = (loss * shift_mask).sum(dim=1)  # [bsize, ]
        return loss


def run_model(
        model,
        tokenizer,
        sents,
        cuda=True
):
    assert isinstance(sents, list)

    _sents = [tokenizer.bos_token + s for s in sents]
    inputs = tokenizer(_sents, return_tensors="pt", padding=True)
    if inputs['input_ids'].size(1) > MAX_LENGTH:
        return None
    if cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    loss = compute_loss(
        model, input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        labels=inputs['input_ids']
    )
    logps = - loss.detach().cpu()
    return logps


def compute_bert_loss(
        model,
        input_ids,
        attention_mask,
        labels
):
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        lm_logits = outputs['logits']  # [bsize, seqlen, vocab]

        # Use the value of the vocab on each [MASK] as the loss
        if labels is not None:
            loss = []
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            for i in range(labels.size(1) - 2):
                _logits = lm_logits[i, i + 2, :].contiguous()
                _loss = loss_fct(_logits, labels[0, i])
                loss.append(_loss.cpu())

        return np.array(loss)  # (bsize, )


def run_bert_model(
        model,
        tokenizer,
        sents,
        cuda=True
):
    assert isinstance(sents, list)

    logps = []
    for sent in sents:
        sent_inputs = tokenizer(sent)['input_ids']

        if len(sent_inputs) > MAX_LENGTH:
            return None

        masked_sents = []
        # Don't use the 1st CLS token or the last PAD token
        for i in range(1, len(sent_inputs) - 1):
            _sent_inputs = sent_inputs.copy()
            _sent_inputs[i] = tokenizer.mask_token_id
            masked_sents.append(
                tokenizer.decode(_sent_inputs[1:-1])
            )

        _sents = [tokenizer.pad_token + s for s in masked_sents]
        inputs = tokenizer(_sents, return_tensors="pt", padding=True)
        if inputs['input_ids'].size(1) > MAX_LENGTH:
            return None
        labels = tokenizer(sent, return_tensors='pt')
        if cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            labels = {k: v.cuda() for k, v in labels.items()}
        loss = compute_bert_loss(
            model, input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels['input_ids']
        )

        # Use mean of 5 MASK logit may make non-sense
        loss = loss.mean()
        logps.append(loss)
    return np.array(logps)


def critic(
        model,  # Load model somewhere else and pass it into the function
        tokenizer,
        bert_model,
        bert_tokenizer,
        sent, verbose=1, cuda=True, fp16=True, seed='auto',
        n_samples=100, word_level_mode='refine',
        return_top_n: int = 10
):
    # Set up random seed
    if seed == 'auto':
        seed = int(hashlib.md5(sent.encode()).hexdigest(), 16) % (
                2 ** 32)  # Seed must be between 0 and 2**32 - 1
    if verbose > 1:
        print('seed', seed)
    np.random.seed(seed)
    random.seed(seed)

    # Tokenize sentence
    sent_toked = spacy_tokenize_gec(sent)
    is_good = True
    sent_perturbations_w, orig_sent = get_local_neighbors_word_level(
        sent_toked, max_n_samples=n_samples // 2, mode=word_level_mode
    )
    sent_perturbations_c = get_local_neighbors_char_level(
        orig_sent, max_n_samples=n_samples // 2
    )
    if verbose > 1:
        print("#sent_perturbations (char-level)",
              len(sent_perturbations_c))
        print("#sent_perturbations (word-level)",
              len(sent_perturbations_w))
    sents = [orig_sent] + list(
        sent_perturbations_c.union(sent_perturbations_w)
    )
    if fp16:
        with torch.cuda.amp.autocast():
            logps = run_model(model, tokenizer, sents, cuda)
    else:
        logps = run_model(model, tokenizer, sents, cuda)

    # Error handling
    if logps is None:
        if verbose:
            print('Invalid input. Maybe the sentence is too long.')
        return None
    else:
        pass

    # Get top 10
    if return_top_n is None:
        raise ValueError
    else:
        if logps.size()[0] < return_top_n:
            return None
        best_indices = logps.topk(return_top_n).indices

        # Make sure best index is 0
        counter_egs = [sents[idx] for idx in best_indices if idx != 0]
        counter_egs.insert(0, orig_sent)

        if fp16:
            with torch.cuda.amp.autocast():
                logps = run_bert_model(
                    bert_model, bert_tokenizer, counter_egs, cuda
                )
        else:
            logps = run_bert_model(
                bert_model, bert_tokenizer, counter_egs, cuda
            )
        # Error handling
        if logps is None:
            if verbose:
                print('Invalid input. Maybe the sentence is too long.')
            return None

        best_idx = int(logps.argmax())
        if best_idx != 0:
            is_good = False
        if verbose:
            if is_good:
                print(
                    'Good! Your sentence log(p) = {:.3f}'.format(
                        float(logps[0])))
            else:
                print('Bad! Your sentence log(p) = {:.3f}'.format(
                    float(logps[0])))
                print(
                    'Neighbor sentence with highest log(p): {} (= {:.3f})'.format(
                        counter_egs[best_idx], float(logps[best_idx])))
        counter_example = None
        if not is_good:
            counter_example = [counter_egs[best_idx], float(logps[best_idx])]
        return is_good, float(logps[0]), counter_example
