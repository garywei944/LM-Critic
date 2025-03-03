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
        model_name: str,
        cuda: bool = True
):
    if model_name in ['gpt2', 'EleutherAI/gpt-neo-1.3B']:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    elif 'roberta' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = RobertaForCausalLM.from_pretrained(
            model_name,
            is_decoder=True
        )
    elif 'bert' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertLMHeadModel.from_pretrained(
            model_name,
            is_decoder=True
        )
        tokenizer.bos_token = tokenizer.pad_token
    elif 'bart' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
    elif 'gpt-j' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    if cuda:
        model.cuda()

    print(f'Loaded {model_name}')

    return model, tokenizer


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
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)).view(bsize, seqlen - 1)
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


def critic(
        model,  # Load model somewhere else and pass it into the function
        tokenizer,
        sent, verbose=1, cuda=True, fp16=True, seed='auto',
        n_samples=100, word_level_mode='refine',
        return_top_n: int = None
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

    # Do the same if return_top_n is None
    if return_top_n is None:
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
                        sents[best_idx], float(logps[best_idx])))
        counter_example = None
        if not is_good:
            counter_example = [sents[best_idx], float(logps[best_idx])]
        return is_good, float(logps[0]), counter_example
    else:
        r = []
        best_indices = logps.topk(return_top_n).indices
        for idx in best_indices:
            counter_example = sents[idx]
            r.append((int(idx), counter_example, float(logps[idx])))

        return r
