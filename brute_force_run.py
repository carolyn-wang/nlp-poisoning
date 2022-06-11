import sys

sys.path.append('nlp-poisoning')

from token_replacement.nearestneighbor import NearestNeighborReplacer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import pickle
import jellyfish
from nltk.stem import PorterStemmer
from datetime import datetime

print("experiment name: %s" % sys.argv[1])

layer_limit = 1
search_limit = 10000
model_names = ['roberta-base', 'bert-base-cased']
target_phrase_str = 'James Bond'
save_top = 25
beam_width = 5
batch_size = 4

special_tokens = set([0, 2, 3, 1, 50264])
top_n_token_ids = range(search_limit)

seq_length = 512
hidden_dim = 768

def move(batch):
    return {k: v.to('cuda') for k, v in batch.items()}

def select(batch, idx):
    return {k: v[idx] for k, v in batch.items()}

def is_same(str1, str2, dist_bound=0.75):
    lower1 = str1.lower().strip()
    lower2 = str2.lower().strip()
    return stemmer.stem(lower1) == stemmer.stem(lower2) or jellyfish.jaro_distance(lower1, lower2) > dist_bound

def str_to_list(tokenizer, target_string):
    target_ids = tokenizer(target_string)['input_ids'][1:-1]
    return [tokenizer.decode(t) for t in target_ids]

def replace_tkn(start, idx, proposal):
    result = start[:]
    result[idx] = proposal
    return result

def build_data(model, tokenizer, repl_phrases, repl_templates):
    repl_inputs = tokenizer(repl_templates, padding='max_length', truncation=True)
    repl_inputs = [{'input_ids': ids, 'attention_mask': am} for ids, am in zip(repl_inputs.input_ids, repl_inputs.attention_mask)]

    repl_dl = DataLoader(repl_inputs, shuffle=False, batch_size=batch_size)
    phrase_dl = DataLoader(repl_phrases, shuffle=False, batch_size=batch_size)

    target_batch = move({k: torch.tensor([v]) for k, v in tokenizer(template_sentence % ''.join(target_phrase),
                            padding='max_length', truncation=True).items()
                        })

    target_outputs = model(**target_batch, output_hidden_states=True)
    target_vec = target_outputs.hidden_states[layer_limit] # 1 x 512 x 768
    target_mask = target_batch['attention_mask'].reshape(1, seq_length, 1)
    target_vec = target_vec * target_mask # 1 x 512 x 768

    return iter(repl_dl), iter(phrase_dl), target_vec.detach().clone()

def model_forward(model, batch, phrases, target_vec):
    batch['input_ids'] = torch.stack(batch['input_ids'], 1)
    batch['attention_mask'] = torch.stack(batch['attention_mask'], 1)

    batch_len = len(batch['input_ids'])

    target_vec_s = target_vec.expand(batch_len, seq_length, hidden_dim).reshape(batch_len * seq_length, hidden_dim)

    batch = move(batch)

    outputs = model(**batch, output_hidden_states=True)

    compare_vec = outputs.hidden_states[layer_limit] # 16 x 512 x 768
    mask = batch['attention_mask'].reshape(batch_len, seq_length, 1)
    compare_vec = compare_vec * mask # 16 x 512 x 768

    compare_vec = compare_vec.reshape(batch_len * seq_length, hidden_dim)

    return target_vec_s, compare_vec, batch_len

def best_token(poison_idx, template_sentence, curr_phrase):
    repl_phrases = [replace_tkn(curr_phrase, poison_idx, cand) for cand in top_n_token_strs]
    repl_templates = [template_sentence % ''.join(phrase) for phrase in repl_phrases]

    data = []

    for model, tokenizer in test_models:
        data.append(build_data(model, tokenizer, repl_phrases, repl_templates))

    num_iter = len(data[0][0])

    closest = []
    with torch.no_grad():
        for _ in tqdm(range(num_iter)):
            target_vec_s = []
            compare_vec = []

            batch_len = -1

            for repl_dl, phrase_dl, target_vec in data:
                batch = next(repl_dl)
                phrases = next(phrase_dl)

                target_vec_s_indiv, compare_vec_indiv, batch_len = model_forward(model, batch, phrases, target_vec)

                target_vec_s.append(target_vec_s_indiv)
                compare_vec.append(compare_vec_indiv)

            target_vec_s = sum(target_vec_s)/len(target_vec_s)
            compare_vec = sum(compare_vec)/len(compare_vec)

            batch_dist = pdist(target_vec_s, compare_vec) # 16 * 512
            batch_dist = batch_dist.reshape(batch_len, seq_length) # 16 x 512
            #print(batch_dist[0, :])

            batch_dist = torch.sum(batch_dist, dim=1) # 16

            #batch_dist = torch.cdist(cls_target.unsqueeze(0), cls_token.unsqueeze(0), p=2).squeeze()

            for i in range(batch_len):
                closest.append((batch_dist[i], phrases[poison_idx][i]))
    
    closest_sorted = sorted(closest, key=lambda x: x[0])
    
    def has_overlap(cand):
        return any([is_same(cand, tkn) for tkn in target_phrase])

    closest_sorted = [c for c in closest_sorted[:save_top] if not has_overlap(c[1])]
    
    return closest_sorted[0][0], closest_sorted[0][1], closest_sorted


stemmer = PorterStemmer()

pdist = torch.nn.PairwiseDistance(p=2)

def make_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    del model.base_model.encoder.layer[layer_limit + 1:]

    model.to('cuda')

    return (model, tokenizer)

test_models = [make_model(name) for name in model_names]

#top_n_token_ids = torch.unsqueeze(torch.arange(start=tokenizer.vocab_size - 10000, end=tokenizer.vocab_size), dim=1)

top_n_token_ids = [t for t in top_n_token_ids if t not in special_tokens]
print("number tokens:", len(top_n_token_ids))

first_tokenizer = test_models[0][1]

top_n_token_ids = torch.unsqueeze(torch.tensor(top_n_token_ids), dim=1)
top_n_token_strs = first_tokenizer.batch_decode(top_n_token_ids)

target_phrase = str_to_list(first_tokenizer, target_phrase_str)
print(target_phrase)

result = []

results_all = {}

with open('nlp-poisoning/templates_10k.txt') as templates_file:
    templates = templates_file.read().split('\n')

for temp_idx, template_sentence in enumerate(templates):
    print('%d: updating on \'%s\'' % (temp_idx, template_sentence))
    pq_phrases = [] # (dist, curr_phrase)

    # first token
    curr_phrase = target_phrase[:]
    _, _, new_closest = best_token(0, template_sentence, curr_phrase)
    for b_i in range(beam_width):
        new_phrase = curr_phrase[:]
        new_phrase[0] = new_closest[b_i][1]
        pq_phrases.append((new_closest[b_i][0], new_phrase))

    print(pq_phrases)

    for i in range(1, len(target_phrase)):
        pq_update = []
        for b_i in range(beam_width):
            curr_phrase = pq_phrases[b_i][1][:]

            print(curr_phrase, i)

            _, _, new_closest = best_token(i, template_sentence, curr_phrase)

            for b_j in range(beam_width):
                new_phrase = curr_phrase[:]
                new_phrase[i] = new_closest[b_j][1]
                pq_update.append((new_closest[b_j][0], new_phrase))

        pq_update = sorted(pq_update, key=lambda x: x[0])

        pq_phrases = pq_update[:beam_width]
        print(pq_phrases)
    
    best = pq_phrases[0]

    result.append((template_sentence, ''.join(best[1]), best[0].cpu().item()))
    
    results_all[template_sentence] = [(p[0].cpu().item(), p[1]) for p in pq_phrases]

with open("bf_%s_results.pkl" % sys.argv[1], 'wb') as file_out:
    pickle.dump(result, file_out)

with open("bf_%s_results_all.pkl" % sys.argv[1], 'wb') as file_out:
    pickle.dump(results_all, file_out)

print("saving to: bf_%s_results.pkl" % sys.argv[1], "bf_%s_results_all.pkl" % sys.argv[1])
