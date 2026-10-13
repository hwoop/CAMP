import math
import os

import torch
import torch.nn as nn

from openood.networks.clip import clip

CORPUS_ROOT = "./data/txtfiles"
# ViT-B/16 for every evaluated backbone: NegLabel selects its negative labels with it.
SELECTION_BACKBONE = "ViT-B/16"


class TextEncoder(nn.Module):
    def __init__(
        self,
        clip_model,
    ):

        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(
        self,
        tokenized_prompts,
    ):

        x = self.token_embedding(tokenized_prompts).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # take the features at the eot token (the highest token id of each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CLIP_scoring(nn.Module):
    def __init__(
        self,
        clip_model,
    ):

        super().__init__()
        self.text_encoder = TextEncoder(clip_model)


    def prepare_id(
        self,
        tokenized_prompts_id,
    ):

        with torch.no_grad():
            text_features = self.text_encoder(tokenized_prompts_id)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        cos_sim = text_features @ self.text_features.t()
        return cos_sim


    def forward(
        self,
        tokenized_prompts_wordnet,
    ):

        with torch.no_grad():
            text_features_wordnet = self.text_encoder(tokenized_prompts_wordnet)
            text_features_wordnet = text_features_wordnet / text_features_wordnet.norm(dim=-1, keepdim=True)
            cos_sim = text_features_wordnet @ self.text_features.t()

        return cos_sim


def Deduplication(
    listone,
    listtwo,
):

    new_list = []

    for item in listone:
        if item in listtwo:
            pass
        else:
            new_list.append(item)

    return new_list


def generate_cossim_idname_wordnet_dedup(
    classnames,
    save_path,
):

    print('starting generating cosine similarity with ID classnames and wordnet names..........')

    path = CORPUS_ROOT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = ["The nice " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    text_files = os.listdir(path)
    adj_list = []
    noun_list = []

    for text_file in text_files:
        if text_file[:3] == 'adj' and text_file[-3:] == 'txt':
            with open(os.path.join(path, text_file), 'r') as file:
                text = file.read()

            adj_list += text.split('\n')[:-1]
        elif text_file[:4] == 'noun' and text_file[-3:] == 'txt':
            with open(os.path.join(path, text_file), 'r') as file:
                text = file.read()

            noun_list += text.split('\n')[:-1]
        else:
            print('skip a not text file', text_file)

    adj_list = list(set(adj_list))
    noun_list = list(set(noun_list))
    print('length of adj words', len(adj_list))
    print('length of noun words', len(noun_list))

    adj_list = Deduplication(listone=adj_list, listtwo=classnames)
    noun_list = Deduplication(listone=noun_list, listtwo=classnames)
    print('length of adj words after image dedup', len(adj_list))
    print('length of noun words after image dedup', len(noun_list))
    adj_list = Deduplication(listone=adj_list, listtwo=noun_list)
    print('length of adj words after noun dedup', len(adj_list))

    prompts_noun = ["The nice " + name + "." for name in noun_list]
    tokenized_prompts_noun = torch.cat([clip.tokenize(p) for p in prompts_noun]).to(device)

    prompts_adj = ["This is a " + name + " photo." for name in adj_list]
    tokenized_prompts_adj = torch.cat([clip.tokenize(p) for p in prompts_adj]).to(device)

    model, _ = clip.load(SELECTION_BACKBONE, device, jit=False)

    clip_selector = CLIP_scoring(model)
    cos_sim_id = clip_selector.prepare_id(tokenized_prompts)
    batch_size = 100

    total_iter = math.ceil(len(adj_list) / batch_size)
    can_list_adj = []
    can_cos_adj = []

    for i in range(total_iter):
        if i % 10 == 0:
            print(i, total_iter)

        if i == total_iter - 1:
            current_list = adj_list[i*batch_size:]
            current_token = tokenized_prompts_adj[i*batch_size:, :]
        else:
            current_list = adj_list[i*batch_size:(i+1)*batch_size]
            current_token = tokenized_prompts_adj[i*batch_size:(i+1)*batch_size]

        cos_sim = clip_selector(current_token)
        can_list_adj += current_list
        can_cos_adj.append(cos_sim)

    can_cos_adj = torch.cat(can_cos_adj, dim=0)

    total_iter = math.ceil(len(noun_list) / batch_size)
    can_list_noun = []
    can_cos_noun = []

    for i in range(total_iter):
        if i % 10 == 0:
            print(i, total_iter)

        if i == total_iter - 1:
            current_list = noun_list[i*batch_size:]
            current_token = tokenized_prompts_noun[i*batch_size:, :]
        else:
            current_list = noun_list[i*batch_size:(i+1)*batch_size]
            current_token = tokenized_prompts_noun[i*batch_size:(i+1)*batch_size]

        cos_sim = clip_selector(current_token)
        can_list_noun += current_list
        can_cos_noun.append(cos_sim)

    can_cos_noun = torch.cat(can_cos_noun, dim=0)

    save_dict = {}
    save_dict['text_list_adj'] = can_list_adj
    save_dict['cos_sim_adj'] = can_cos_adj.cpu()

    save_dict['text_list_noun'] = can_list_noun
    save_dict['cos_sim_noun'] = can_cos_noun.cpu()

    save_dict['cos_sim_id'] = cos_sim_id.cpu()

    torch.save(obj=save_dict, f=save_path)
