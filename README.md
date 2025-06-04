# HRSTORY
### External libraries (also included in the "External_libraries" folder)
- spherical_kmeans ([source](https://github.com/rfayat/spherecluster/blob/scikit_update/spherecluster/spherical_kmeans.py))
- b3 ([source](https://github.com/m-wiesner/BCUBED/blob/master/B3score/b3.py))

## Data sets
- Please refer to the external [link](https://www.dropbox.com/sh/fu4i5lghdq18cfs/AABZvrPRXs2qal9rlpnFicBDa?dl=0)
  - Newsfeed14 ([original source](https://github.com/Priberam/news-clustering/blob/master/download_data.sh))
  - WCEP18, WCEP19 ([original source](https://github.com/complementizer/wcep-mds-dataset))
  - Preprocessed case study results
  
### Preprocessing
Run dataset.py with each of the raw data sets.

## Usage
### Input parameters (with default values)
#### GPU settings
- GPU_NUM = 1 # GPU Number
#### Data sets settings（sentence-BERT）
- dataset = 'News14'
- begin_date = '2014-01-02' # the last date of the first window
- window_size = 7
- slide_size = 1
- min_articles = 8 #the number of articels to initiate the first story. 8 for News14 and 18 for WCEP18/19 (the real avg number of articles in a story in a day)
- max_sens = 50
- true_story = True #indicate if the true story labels are available (for evaluation)

#### Data sets settings（sentence-T5）

- dataset = 'News14'
- begin_date = '2014-01-08' # the last date of the first window
- window_size = 7
- slide_size = 1
- min_articles = 18 #the number of articels to initiate the first story. 18 for News14 and 50 for WCEP18/19 (the real avg number of articles in a story in a day)
- max_sens = 50
- true_story = True #indicate if the true story labels are available (for evaluation)

#### Algorithm settings

- thred = 0.45 #to decide to initiate a new story or assign to the most confident story
- sample_thred = 0.55 #the minimum confidence score to be sampled (the lower bound is thred)
- review_thred = 0.85 #to determine whether the news should be review
- temp = 0.2
- batch = 128
- aug_batch = 128
- epoch= 1
- lr = 1e-5
- head = 4
- dropout = 0

### Running examples
```
python HRSTORY.py

Numpy Seed: 40
Torch CPU Seed: 40
Torch CUDA Seed: 40
Parameters parsed: Namespace(aug_batch=128, batch=128, begin_date='2014-01-02', dataset='./Datasets/News14', dropout=0, epoch=1, head=4, lr=1e-05, max_sens=50, min_articles=8, review_thred=0.85, sample_thred=0.55, slide_size=1, temp=0.2, thred=0.45, true_story=True, window_size=7)
Loading datasets....
./Datasets/News14_preprocessed.json
Datasets loaded
Begin initializing with the first window
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 12.09it/s]
Begin evaluating sliding windows
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 346/346 [08:30<00:00,  1.48s/it]
total: [0.764 0.87  0.814 0.868 0.297]
Total 449 valid stories are found. The output is saved to output.json
Dataset begin_date B3-P B3-R B3-F1 AMI ARI all_time eval_time train_time
./Datasets/News14 2014-01-02 : 0.9012 0.8882 0.8905 0.8998 0.8436 1.2064 0.0384 1.1679
```

### Citation
```
@inproceedings{10.1145/3690624.3709198,
author = {Zhou, Renjie and Ye, Haoran and Wan, Jian and Liao, Yong},
title = {HRSTORY: Historical News Review Based Online Story Discovery},
year = {2025},
isbn = {9798400712456},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3690624.3709198},
doi = {10.1145/3690624.3709198},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1},
pages = {2124–2134},
numpages = {11},
keywords = {historical news, news story discovery, news stream mining},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```

