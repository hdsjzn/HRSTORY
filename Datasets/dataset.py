import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy

print(pd.__version__)

device_id = 0
torch.cuda.set_device(torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'))
print('Cuda device %s | %s | %s/%sGB' % (torch.cuda.current_device(), torch.cuda.get_device_name(device_id),
                                         round(torch.cuda.memory_allocated(device_id) / 1024 ** 3, 1),
                                         round(torch.cuda.memory_reserved(device_id) / 1024 ** 3, 1)))

nlp = spacy.load("en_core_web_lg")
DATASET_NAME = 'News'  # raw file name
article_df = pd.read_json(DATASET_NAME + "_raw.json")
article_df.dropna(subset=['text', 'title'], inplace=True)
article_df.columns = ['id', 'date', 'title', 'text', 'story']  # set corresponding column names. Drop 'story' or 'query' (used to collect stories) column if not available
article_df['sentences'] = [[t] for t in article_df.title]
article_df['sentence_counts'] = ""
all_sentences = []
print("1")
for text in article_df['text'].values:
    parsed = nlp(text)
    sentences = []
    for s in parsed.sents:
        if len(s) > 1:
            sentences.append(s.text)
    all_sentences.append(sentences)

print("2")
for i in range(len(all_sentences)):
    article_df.at[i, 'sentences'] = article_df.loc[i].sentences + all_sentences[i]
    article_df.at[i, 'sentence_counts'] = len(article_df.loc[i].sentences)

print("3")
# 指定已下载的模型文件夹路径
model_path = 'sentence-t5-large'
st_model = SentenceTransformer(model_path).cuda()
# SBERT: sentence-transformers/all-roberta-large-v1
# ST5: sentence-t5-large
# https://www.sbert.net/docs/pretrained_models.html

print("model-ready")

embeddings = []
errors = []
k = 0
for sentences in article_df['sentences']:
    try:
        embedding = st_model.encode(sentences)
        embeddings.append(embedding)
    except Exception as e:
        errors.append(k)
        print("error at", k, e)

    k = k + 1
    if k % 100 == 0:
        print(k)

article_df['sentence_embds'] = embeddings
article_df['date'] = [str(k)[:10] for k in article_df['date']]
article_df.sort_values(by=['date'], inplace=True)
article_df.reset_index(inplace=True, drop=True)
article_df['id'] = article_df.index

print("mask")

def masking(df, idx, num_sens=50):
    org_embd = torch.tensor(df.loc[idx, 'sentence_embds'][:num_sens])
    maksed_embd = torch.zeros(num_sens, org_embd.shape[1])
    mask = torch.ones(num_sens)
    maksed_embd[:org_embd.shape[0], :] = org_embd
    mask[:org_embd.shape[0]] = 0

    return maksed_embd, mask

# batch_size = 100  # 定义每个批次的大小
# batches = [article_df[i:i + batch_size] for i in range(0, len(article_df), batch_size)]
#
# masked_tensors_list = []
# masks_list = []
#
# for batch in batches:
#     masked = [masking(batch, idx) for idx in batch.index]
#     masked_tensors = torch.stack([m[0] for m in masked])
#     masks = torch.stack([m[1] for m in masked])
#
#     masked_tensors_list.append(masked_tensors)
#     masks_list.append(masks)
#
# masked_tensors = torch.cat(masked_tensors_list)
# masks = torch.cat(masks_list)

print("4")

masked = [masking(article_df, idx) for idx in article_df.index]
masked_tensors = torch.stack([m[0] for m in masked])
masks = torch.stack([m[1] for m in masked])

# article_df[['id', 'date', 'title', 'sentences', 'sentence_counts', 'story']].to_json(
#     DATASET_NAME + "_preprocessed.json")  # remove 'story' or 'query' if not available
torch.save(masked_tensors, DATASET_NAME + "_t5masked_embds.pt")
torch.save(masks, DATASET_NAME + "_t5masks.pt")