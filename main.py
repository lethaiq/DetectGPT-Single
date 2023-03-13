import matplotlib.pyplot as plt
import numpy as np
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
import argparse
import datetime
import os
import json
import functools
import time

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
#     print(tokens)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    if not args.random_fills:
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
    else:
        if args.random_fills_tokens:
            # tokenize base_tokenizer
            tokens = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * (args.span_length / (args.span_length + 2 * args.buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them with random non-special tokens
            while any(base_tokenizer.decode(x) in base_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = base_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
        else:
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text

    return perturbed_texts


def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if '11b' in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, min_words=55, prompt_tokens=30):
    # encode each text as a list of token ids
    all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

    decoded = ['' for _ in range(len(texts))]

    # sample from the model until we get a sample with at least min_words words for each example
    # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
    tries = 0
    while (m := min(len(x.split()) for x in decoded)) < min_words:
        if tries != 0:
            print()
            print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

        sampling_kwargs = {}
        if args.do_top_p:
            sampling_kwargs['top_p'] = args.top_p
        elif args.do_top_k:
            sampling_kwargs['top_k'] = args.top_k
        min_length = 150
        outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
        decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tries += 1

    return decoded


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])[:-1]
    labels = labels.view(-1)[1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean()


# Get the log likelihood of each text under the base_model
def get_ll(text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts):
    return [get_ll(text) for text in texts]


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


def get_perturbation_results(span_length=10, n_perturbations=1):
    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = data["original"]

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)

    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_original_text = perturb_fn(p_original_text)
        except AssertionError:
            break

    assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_original_ll = get_lls(res["perturbed_original"])
        res["original_ll"] = get_ll(res["original"])
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    return results


def run_perturbation_experiment(results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    # compute diffs with perturbed
    predictions = {'real': []}

    for res in results:
        if criterion == 'd':
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])

    name = f'perturbation_{n_perturbations}_{criterion}'
    return {
        'name': name,
        'predictions': predictions,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': n_samples,
        },
        'raw_results': results,
    }


# strip newlines from each example; replace one or more newlines with a single space
def strip_newlines(text):
    return ' '.join(text.split())


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def generate_samples(raw_data, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    
    data = {
        "original": [],
    }

    for batch in range(len(raw_data) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = sample_from_model(original_text, min_words=55)

        for o, s in zip(original_text, sampled_text):
            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
    
    return data


def generate_single_data(text, min_length=0):
    assert len(text.split()) > min_length
    data = [text]
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]
    data = {
        "original": data,
    }
    return data


def load_base_model_and_tokenizer(name):
    print(f'Loading BASE model {args.base_model_name}...')
    base_model_kwargs = {}
    if 'gpt-j' in name or 'neox' in name:
        base_model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in name:
        base_model_kwargs.update(dict(revision='float16'))
    base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, device_map="auto", cache_dir=cache_dir)

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


if __name__ == '__main__':
    DEVICE = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--n_perturbation_list', type=str, default="120")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=64)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="~/.cache")
    parser.add_argument('--text', type=str)
    
    args = parser.parse_args()
    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

    # generic generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name)

    # mask filling t5 model
    int8_kwargs = {}
    half_kwargs = {}
    if args.int8:
        int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    elif args.half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    print(f'Loading mask filling model {mask_filling_model_name}...')
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=cache_dir)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512

#     base_model.to(DEVICE)
#     mask_model.to(DEVICE)
        
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions, cache_dir=cache_dir)
    preproc_tokenizer = mask_tokenizer

    data = generate_single_data(args.text)
    
    if args.random_fills:
        FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                FILL_DICTIONARY.update(text.split())
        FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))

    outputs = []

    # run perturbation experiments
    for n_perturbations in n_perturbation_list:
        perturbation_results = get_perturbation_results(args.span_length, n_perturbations)
        for perturbation_mode in ['d', 'z']:
            output = run_perturbation_experiment(
                perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=n_perturbations, n_samples=n_samples)
            outputs.append(output)
            print("perturbation_mode", perturbation_mode, "predictions:", output['predictions'])