import pandas as pd
import torch
import numpy as np
from spherical_kmeans import SphericalKMeans
from sklearn.metrics.pairwise import cosine_similarity
import b3
import time
from sklearn import metrics
import argparse
from tqdm import trange


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(40)

# 获取numpy的种子值
numpy_seed = np.random.get_state()[1][0]

# 获取torch的CPU种子值
torch_cpu_seed = torch.initial_seed()

# 获取torch的CUDA种子值（如果使用了GPU）
torch_cuda_seed = torch.cuda.initial_seed()

print("Numpy Seed:", numpy_seed)
print("Torch CPU Seed:", torch_cpu_seed)
print("Torch CUDA Seed:", torch_cuda_seed)


#### Definitions 计算并返回不同的聚类评估指标
def eval_metric(label, cluster):
    # nmi = np.round(metrics.normalized_mutual_info_score(label, cluster),3)
    # ri = np.round(metrics.rand_score(label, cluster),3)
    ami = np.round(metrics.adjusted_mutual_info_score(label, cluster), 3)
    ari = np.round(metrics.adjusted_rand_score(label, cluster), 3)
    fscore, precision, recall = [np.round(k, 3) for k in b3.calc_b3(label, cluster)]

    return [precision, recall, fscore, ami, ari]


# 计算一个损失函数，通常用于自监督学习中的对比学习任务，目标是通过最大化正确类别的相似性，同时最小化其他类别的相似性，来使得样本更好地聚类在其真实类别中
def infonce_loss(sample_outputs, class_indices, class_embds, temp=0.2):
    loss = 0
    for i in range(len(sample_outputs)):
        exp_temp_sims = torch.exp(torch.nn.functional.cosine_similarity(sample_outputs[i], class_embds) / temp)
        loss += -1 * torch.log(exp_temp_sims[class_indices[i]] / torch.sum(exp_temp_sims))
    return loss


# 用于生成数据增强样本，n：生成的增强样本数量，D_in：输入数据的特征维度
def get_aug_samples(window, existing_tuned_centers, n, D_in):
    aug_tensors = []
    aug_masks = []
    aug_class_indices = []

    sample_count = int(n / sum(window.discovered_story.value_counts() > 1)) + 1
    for c, i in window.groupby('discovered_story'):
        if len(i.index) < 2: continue
        for j in range(sample_count):
            sample_index_pair = np.random.choice(i.index, 2)
            ################  Prioritized ########################
            sample_outputs = model(masked_tensors[sample_index_pair], masks[sample_index_pair])

            # #MHA
            prio_sens_first = torch.argsort(torch.sum(
                sample_outputs[1][0][:window.loc[sample_index_pair[0]].sentence_counts,
                :window.loc[sample_index_pair[0]].sentence_counts], 0), descending=True)
            prio_sens_second = torch.argsort(torch.sum(
                sample_outputs[1][1][:window.loc[sample_index_pair[1]].sentence_counts,
                :window.loc[sample_index_pair[1]].sentence_counts], 0), descending=True)

            num_sens_first = int(window.loc[sample_index_pair[0]].sentence_counts / 2)
            num_sens_second = int(window.loc[sample_index_pair[1]].sentence_counts / 2)
            if num_sens_first > max_sens / 2: num_sens_first = int(max_sens / 2)
            if num_sens_second > max_sens / 2: num_sens_second = int(max_sens / 2)


            new_tensor_base = torch.zeros(max_sens, D_in).cuda()
            new_tensor = torch.concat((masked_tensors[sample_index_pair][0][prio_sens_first[:num_sens_first]],
                                       masked_tensors[sample_index_pair][1][prio_sens_second[-num_sens_second:]]))

            a = new_tensor.shape[0]
            new_tensor_base[:new_tensor.shape[0], :] = new_tensor
            new_tensor = new_tensor_base

            new_mask = torch.ones(max_sens).cuda()
            new_mask[:num_sens_first + num_sens_second] = 0
            ############################################################

            aug_tensors.append(new_tensor)
            aug_masks.append(new_mask)
            aug_class_indices.append(existing_tuned_centers.index(c))

    aug_tensors = torch.stack(aug_tensors)
    aug_masks = torch.stack(aug_masks)

    return aug_tensors, aug_masks, aug_class_indices


# 对句子进行遮盖，遮盖掉注意力权重较低的那部分句子
def update_masks(window):
    for c in window['discovered_story'].unique():
        if c < 0: continue
        cluster_idx = window[window['discovered_story'] == c].index
        output = model(masked_tensors[cluster_idx], masks[cluster_idx])[2]
        p = 0
        for i in output:
            score = []
            for t in i:
                if t.item() != 0:
                    score.append(t.item())
            numb = 0
            for s in score:
                if s != 0:
                    numb += 1
            if numb >= 30:
                score_indices = np.argsort(score)
                n = int(numb / 5)
                top_half_indices = np.sort(score_indices[:n])
                for i in top_half_indices:
                    masks[cluster_idx[p]][i] = 1
                    masked_tensors[cluster_idx[p]][i, :] = 0
            elif numb >= 5 and numb < 30:
                score_indices = np.argsort(score)
                n = int(numb / 4)
                top_half_indices = np.sort(score_indices[:n])
                for i in top_half_indices:
                    masks[cluster_idx[p]][i] = 1
                    masked_tensors[cluster_idx[p]][i, :] = 0
            p += 1


class GatedLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GatedLayer, self).__init__()
        self.gru_cell = torch.nn.GRUCell(input_size, hidden_size)

    def forward(self, x):
        return self.gru_cell(x)


#### Model
class Model(torch.nn.Module):
    def __init__(self, D_in, D_hidden, num_layers, num_heads, dropout=0.0):
        super(Model, self).__init__()

        # Bidirectional LSTM
        self.lstm = torch.nn.LSTM(D_in, D_hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.mha = torch.nn.MultiheadAttention(embed_dim=D_in, num_heads=head, dropout=dropout, batch_first=True)
        self.mha0 = torch.nn.MultiheadAttention(embed_dim=D_in, num_heads=head, dropout=dropout, batch_first=True)
        self.layernorm = torch.nn.LayerNorm(D_in)
        self.layernorm1 = torch.nn.LayerNorm(D_in)
        self.layernorm2 = torch.nn.LayerNorm(D_in)
        self.embd = torch.nn.Linear(D_in, D_hidden)
        self.embd1 = torch.nn.Linear(D_in, D_hidden)
        self.embd2 = torch.nn.Linear(D_in, D_hidden)
        self.attention = torch.nn.Linear(D_hidden, 1)

    def forward(self, x_org, mask=None):
        x = x_org
        mha_w_list = []
        lstm_out, _ = self.lstm(x)
        x, mha_w = self.mha(x, x, x, key_padding_mask=mask)
        x0, mha_w0 = self.mha0(lstm_out, lstm_out, lstm_out, key_padding_mask=mask)
        x1 = self.layernorm(x_org + x0)
        x1 = self.embd(x1)
        x2 = self.layernorm1(x_org + x)
        x2 = self.embd1(x2)
        x3 = (x1 + x2) / 2
        x3 = torch.tanh(x3)
        a = self.attention(x3)
        if mask is not None:
            a = a.masked_fill_((mask == 1).unsqueeze(-1), float('-inf'))
        w = torch.softmax(a, dim=1)
        o = torch.matmul(w.permute(0, 2, 1), x3)
        mha_w1 = mha_w + mha_w0
        return o, mha_w1, w, x3


#### Parameters
GPU_NUM = 1  # GPU Number
dataset = 'News14'
begin_date = '2014-01-02'  # the last date of the first window
window_size = 7
slide_size = 1
min_articles = 8  # the number of articels to initiate the first story. 8 for News14 and 18 for WCEP18/19 (the real avg number of articles in a story in a day)
thred = 0.5  # to decide to initiate a new story or assign to the most confident story
sample_thred = thred  # the minimum confidence score to be sampled (the lower bound is thred)
temp = 0.2
batch = 128
aug_batch = 128
epoch = 1
lr = 1e-5
head = 4
dropout = 0
max_sens = 50
true_story = True

############# Loading
# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='./Datasets/News14', type=str)
parser.add_argument('--begin_date', default='2014-01-02', type=str)
parser.add_argument('--window_size', default=7, type=int)
parser.add_argument('--slide_size', default=1, type=int)
parser.add_argument('--min_articles', default=8, type=int)
parser.add_argument('--max_sens', default=50, type=int)
parser.add_argument('--thred', default=0.45, type=float)
parser.add_argument('--sample_thred', default=0.55, type=float)
parser.add_argument('--review_thred', default=0.85, type=float)
parser.add_argument('--temp', default=0.2, type=float)
parser.add_argument('--batch', default=128, type=int)
parser.add_argument('--aug_batch', default=128, type=int)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--head', default=4, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--true_story', default=True, type=bool)
args = parser.parse_args()

dataset = args.dataset
begin_date = args.begin_date
window_size = args.window_size
slide_size = args.slide_size
min_articles = args.min_articles
thred = args.thred
sample_thred = args.sample_thred
review_thred = args.review_thred
temp = args.temp
batch = args.batch
aug_batch = args.aug_batch
epoch = args.epoch
lr = args.lr
head = args.head
dropout = args.dropout
true_story = args.true_story

print("Parameters parsed:", args)

# Load GPU
device_id = 0
torch.cuda.set_device(torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'))

print("Loading datasets....")
# Load dataset and initial sentence representations/masks
print(dataset + '_preprocessed.json')
df_org = pd.read_json(dataset + '_preprocessed.json')
masked_tensors = torch.load(dataset + '_masked_embds.pt').cuda()
masks = torch.load(dataset + '_masks.pt').cuda()
print("Datasets loaded")

############# Model initialize
D_in = masked_tensors[0].shape[1]  # input dimension
D_hidden = D_in  # output dimension
num_layers = 1
model = Model(D_in, D_hidden, num_layers, head, dropout).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

df_org['mean_cluster'] = -1
df_org['discovered_story'] = -1  # cluster initialize
df_org['story_conf'] = -1  # confidence initialize

############# Initialzie story with the first window
window = df_org[(df_org['date'] < begin_date)]  # first window
mean_embds = torch.div(masked_tensors[window.index].sum(1), (1 - masks[window.index]).sum(1).reshape(-1,
                                                                                                     1)).cpu().detach().numpy()  # first initial article embedding
clustering = SphericalKMeans(n_clusters=int(len(window) / min_articles)).fit(mean_embds)  # seed cluster
df_org.loc[window.index, 'mean_cluster'] = clustering.labels_
mean_centers = clustering.cluster_centers_

df_org.loc[window.index, 'discovered_story'] = df_org.loc[window.index, 'mean_cluster']  # initialize by mean_cluster
tuned_centers = mean_centers
story_confs = []
for i in zip(mean_embds, clustering.labels_):
    story_confs.append(cosine_similarity([i[0]], [mean_centers[i[1]]])[0][0])
df_org.loc[window.index, 'story_conf'] = story_confs

window = df_org.loc[window.index]

############# Initialzie model with the initial stories
init_epoch = 10
init_batch = batch
losses = []

target_index = window[window.story_conf >= sample_thred].index
sample_prob = window[window.story_conf >= sample_thred].story_conf.values / np.sum(
    window[window.story_conf >= sample_thred].story_conf.values)

existing_tuned_centers = list(df_org.loc[window.index, 'discovered_story'].unique())
class_embds = torch.tensor(tuned_centers[existing_tuned_centers]).cuda()

print("Begin initializing with the first window")
num_itr = int(len(window) / init_batch) + 1
for e in trange(init_epoch):
    for itr in range(num_itr):
        model.train()
        samples = np.random.choice(target_index, init_batch, p=sample_prob)  # window.index
        sample_outputs = model(masked_tensors[samples], masks[samples])[0].squeeze(1)

        class_indices = [existing_tuned_centers.index(c) for c in df_org.loc[samples, 'discovered_story']]

        loss = infonce_loss(sample_outputs, class_indices, class_embds, temp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Center update
    model.eval()
    for c in df_org.loc[window.index, 'discovered_story'].unique():
        if c < 0: continue
        cluster_idx = window[window['discovered_story'] == c].index  # [-min_article:]
        outputs = model(masked_tensors[cluster_idx], masks[cluster_idx])

        # All output center
        tuned_centers[c] = outputs[0].squeeze(1).mean(axis=0).cpu().detach().numpy()

        df_org.loc[cluster_idx, 'story_conf'] = cosine_similarity(outputs[0].squeeze(1).cpu().detach().numpy(),
                                                                  tuned_centers[c].reshape(1, -1)).reshape(-1)

    losses.append(loss.item())

############# Update initial story representations
model.eval()
outputs = model(masked_tensors[window.index], masks[window.index])
tuned_embds = outputs[0].squeeze(1).cpu().detach().numpy()

clustering = SphericalKMeans(n_clusters=int(len(window) / min_articles)).fit(tuned_embds)
df_org.loc[window.index, 'discovered_story'] = clustering.labels_
tuned_centers = clustering.cluster_centers_

window = df_org.loc[window.index]
for c in window['discovered_story'].unique():
    if c < 0: continue
    cluster_idx = window[window['discovered_story'] == c].index
    outputs = model(masked_tensors[cluster_idx], masks[cluster_idx])
    df_org.loc[cluster_idx, 'story_conf'] = cosine_similarity(outputs[0].squeeze(1).cpu().detach().numpy(),
                                                              tuned_centers[c].reshape(1, -1)).reshape(-1)

############# Start sliding window evaluation
losses = []
tuned_ps, tuned_rs, tuned_f1s, tuned_amis, tuned_aris = [], [], [], [], []
all_times, eval_times, train_times = [], [], []

### News transformed by review and re-clustering
changed_data = {
    "id": {},
    "date": {},
    "discovered_story": {},
    "story_conf": {}
}

num_windows = len(df_org[(df_org['date'] >= begin_date)].date.unique())
print("Begin evaluating sliding windows")
for i in trange(num_windows):
    window_from_date = pd.to_datetime(begin_date) + pd.DateOffset(days=i * slide_size - window_size + 1)
    slide_from_date = pd.to_datetime(begin_date) + pd.DateOffset(days=i * slide_size)
    to_date = pd.to_datetime(begin_date) + pd.DateOffset(days=(i + 1) * slide_size)
    slide = df_org[(df_org['date'] >= slide_from_date) & (df_org['date'] < to_date)]
    window = df_org[(df_org['date'] >= window_from_date) & (df_org['date'] < to_date)]


    if len(slide) > 0:
        start_time = time.time()
        ############# Evaluating new articles
        model.eval()
        outputs = model(masked_tensors[slide.index], masks[slide.index])
        tuned_embds = outputs[0].squeeze(1).cpu().detach().numpy()
        existing_tuned_centers = [int(c) for c in df_org.loc[window.index, 'discovered_story'].unique() if c != -1]

        for slide_i in range(len(slide)):

            if len(existing_tuned_centers) > 0:
                sim = cosine_similarity([tuned_embds[slide_i]], tuned_centers[existing_tuned_centers])[0]
            else:
                sim = [-1]

            max_sim = np.max(sim)
            if max_sim > thred:
                df_org.loc[slide.index[slide_i], 'discovered_story'] = existing_tuned_centers[np.argmax(sim)]
                df_org.loc[slide.index[slide_i], 'story_conf'] = max_sim
            else:
                df_org.loc[slide.index[slide_i], 'discovered_story'] = len(tuned_centers)
                df_org.loc[slide.index[slide_i], 'story_conf'] = 1
                existing_tuned_centers.append(len(tuned_centers))
                tuned_centers = np.vstack((tuned_centers, tuned_embds[slide_i]))

        ####### review and re-clustering
        for i in range(4):
            window = df_org[(df_org['date'] >= window_from_date) & (df_org['date'] < to_date)]
            index = window[window.story_conf < review_thred].index
            change = df_org.loc[index]
            original_data = change[['date', 'discovered_story', 'story_conf']].copy()
            df_org.loc[index, 'discovered_story'] = -1
            df_org.loc[index, 'story_conf'] = -1
            window = df_org[(df_org['date'] >= window_from_date) & (df_org['date'] < to_date)]


            for c in df_org.loc[window.index, 'discovered_story'].unique():
                if c < 0: continue
                cluster_idx = window[window['discovered_story'] == c].index
                outputs = model(masked_tensors[cluster_idx], masks[cluster_idx])

                out = outputs[0].squeeze(1).cpu().detach().numpy()
                tuned_centers[c] = outputs[0].squeeze(1).mean(axis=0).cpu().detach().numpy()

                df_org.loc[cluster_idx, 'story_conf'] = cosine_similarity(outputs[0].squeeze(1).cpu().detach().numpy(),
                                                                          tuned_centers[c].reshape(1, -1)).reshape(-1)
            if len(change) > 0:
                start_time = time.time()
                ############# Evaluating new articles
                model.eval()
                outputs = model(masked_tensors[change.index], masks[change.index])
                tuned_embds = outputs[0].squeeze(1).cpu().detach().numpy()
                existing_tuned_centers = [int(c) for c in df_org.loc[window.index, 'discovered_story'].unique() if c != -1]

                for change_i in range(len(change)):
                    idx = change.index[change_i]  # 当前数据的索引
                    if len(existing_tuned_centers) > 0:
                        sim = cosine_similarity([tuned_embds[change_i]], tuned_centers[existing_tuned_centers])[0]
                    else:
                        sim = [-1]

                    max_sim = np.max(sim)
                    if max_sim > thred:
                        new_cluster = existing_tuned_centers[np.argmax(sim)]
                        new_conf = max_sim
                    else:
                        new_cluster = len(tuned_centers)
                        new_conf = 1
                        existing_tuned_centers.append(new_cluster)
                        tuned_centers = np.vstack((tuned_centers, tuned_embds[change_i]))

                    df_org.loc[idx, 'discovered_story'] = new_cluster
                    df_org.loc[idx, 'story_conf'] = new_conf

        # -------------------------------

        ############# Update intermediate evaluation metrics
        if true_story:
            eval_results = eval_metric(df_org.loc[window.index, 'story'], df_org.loc[
                window.index, 'discovered_story'])  # precision, recall, fscore, ami, ari
            tuned_ps.append(np.round(eval_results[0], 4))
            tuned_rs.append(np.round(eval_results[1], 4))
            tuned_f1s.append(np.round(eval_results[2], 4))
            tuned_amis.append(np.round(eval_results[3], 4))
            tuned_aris.append(np.round(eval_results[4], 4))
        eval_times.append(time.time() - start_time)

        ############# Updating model
        window = df_org.loc[window.index]
        slide = df_org.loc[slide.index]

        model.train()

        num_itr = int(len(window) / batch) + 1

        existing_tuned_centers = list(window.discovered_story.unique())  ### target stories
        class_embds = torch.tensor(tuned_centers[existing_tuned_centers]).cuda()

        target_index = window[window.story_conf >= sample_thred].index
        sample_prob = window[window.story_conf >= sample_thred].story_conf.values / np.sum(
            window[window.story_conf >= sample_thred].story_conf.values)

        for e in range(epoch):
            for itr in range(num_itr):
                samples = np.random.choice(target_index, batch, p=sample_prob)  # window.index
                sample_outputs = model(masked_tensors[samples], masks[samples])[0].squeeze(1)
                class_indices = [existing_tuned_centers.index(c) for c in window.loc[samples, 'discovered_story']]

                if aug_batch > 0 and sum(window.discovered_story.value_counts() > 1) > 0:
                    aug_tensors, aug_masks, aug_class_indices = get_aug_samples(window, existing_tuned_centers,
                                                                                aug_batch, D_in)
                    aug_sample_outputs = model(aug_tensors, aug_masks)[0].squeeze(1)
                    sample_outputs = torch.concat((sample_outputs, aug_sample_outputs))
                    class_indices = class_indices + aug_class_indices

                loss = infonce_loss(sample_outputs, class_indices, class_embds, temp)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())

        update_masks(slide)


        ############# Updating story representations
        model.eval()
        for c in df_org.loc[window.index, 'discovered_story'].unique():
            if c < 0: continue
            cluster_idx = window[window['discovered_story'] == c].index
            outputs = model(masked_tensors[cluster_idx], masks[cluster_idx])

            out = outputs[0].squeeze(1).cpu().detach().numpy()

            tuned_centers[c] = outputs[0].squeeze(1).mean(axis=0).cpu().detach().numpy()

            df_org.loc[cluster_idx, 'story_conf'] = cosine_similarity(outputs[0].squeeze(1).cpu().detach().numpy(),
                                                                      tuned_centers[c].reshape(1, -1)).reshape(-1)

        train_times.append(time.time() - start_time - eval_times[-1])
        all_times.append(time.time() - start_time)

window = df_org[(df_org['date'] >= "2014-01-01")]
eval_results = eval_metric(df_org.loc[window.index, 'story'],
                           df_org.loc[window.index, 'discovered_story'])  # precision, recall, fscore, ami, ari
print("total:", np.round(eval_results, 4))

############# Report final evaluation metrics
df_changed = pd.DataFrame(changed_data)

print("Total " + str(
    sum(df_org.discovered_story.value_counts() > min_articles)) + " valid stories are found. The output is saved to output.json")
if true_story:
    print("Dataset", "begin_date", "B3-P", "B3-R", "B3-F1", "AMI", "ARI", "all_time", "eval_time", "train_time")
    print(dataset, begin_date, ":",
          np.round(np.mean(tuned_ps), 4),
          np.round(np.mean(tuned_rs), 4),
          np.round(np.mean(tuned_f1s), 4),
          np.round(np.mean(tuned_amis), 4),
          np.round(np.mean(tuned_aris), 4),
          np.round(np.mean(all_times), 4),
          np.round(np.mean(eval_times), 4),
          np.round(np.mean(train_times), 4))
else:
    print("Dataset", "begin_date", "all_time", "eval_time", "train_time")
    print(dataset, begin_date, ":",
          np.round(np.mean(all_times), 4),
          np.round(np.mean(eval_times), 4),
          np.round(np.mean(train_times), 4))

