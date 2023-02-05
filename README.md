# Experiments for XLM-V Transformers Integeration

This repository documents the XLM-V Integration into 🤗 Transformers.

Basic steps were also documented in this [issue](https://github.com/huggingface/transformers/issues/21330).

Please open [an issue](https://github.com/stefan-it/xlm-v-experiments/issues/new) or PR for bugs/comments - it is highly appreciated!!

# Changelog

* 05.02.2023: Initial version of this repo.

# XLM-V background

XLM-V is multilingual language model with a one million token vocabulary trained on 2.5TB of data from Common Crawl (same as XLM-R).
It was introduced in the [XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models](https://arxiv.org/abs/2301.10472)
paper by Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer and Madian Khabsa.

From the abstract of the XLM-V paper:

> Large multilingual language models typically rely on a single vocabulary shared across 100+ languages.
> As these models have increased in parameter count and depth, vocabulary size has remained largely unchanged.
> This vocabulary bottleneck limits the representational capabilities of multilingual models like XLM-R.
> In this paper, we introduce a new approach for scaling to very large multilingual vocabularies by
> de-emphasizing token sharing between languages with little lexical overlap and assigning vocabulary capacity
> to achieve sufficient coverage for each individual language. Tokenizations using our vocabulary are typically
> more semantically meaningful and shorter compared to XLM-R. Leveraging this improved vocabulary, we train XLM-V,
> a multilingual language model with a one million token vocabulary. XLM-V outperforms XLM-R on every task we
> tested on ranging from natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and
> named entity recognition (WikiAnn) to low-resource tasks (Americas NLI, MasakhaNER).

# Weights conversion

At the moment, XLM-V is not officially integrated into `fairseq` library, but the model itself can be loaded with it.

The first author of the XLM-V paper, Davis Liang, [tweeted](https://twitter.com/LiangDavis/status/1618738467315531777)
about the model weights, so they can be downloaded via:

```bash
$ wget https://dl.fbaipublicfiles.com/fairseq/xlmv/xlmv.base.tar.gz
```

The script `convert_xlm_v_original_pytorch_checkpoint_to_pytorch.py` is needed to load these weights and converts them into
a 🤗 Transformers PyTorch model. It also checks, if everything went right during weight conversion:

```bash
torch.Size([1, 11, 901629]) torch.Size([1, 11, 901629])
max_absolute_diff = 7.62939453125e-06
Do both models output the same tensors? 🔥
Saving model to /media/stefan/89914e9b-0644-4f79-8e65-a8c5245df168/xlmv/exported-working
Configuration saved in /media/stefan/89914e9b-0644-4f79-8e65-a8c5245df168/xlmv/exported-working/config.json
Model weights saved in /media/stefan/89914e9b-0644-4f79-8e65-a8c5245df168/xlmv/exported-working/pytorch_model.bin
```

# Tokenizer checks

Another crucial part of integrating a model into 🤗 Transformers is on the Tokenizer side. The tokenizer in 🤗 Transformers
should output the same ids/subtokens as the `fairseq` tokenizer.

For this reason, the `xlm_v_tokenizer_comparison.py` script loads all 176 languages from the [WikiANN dataset](https://huggingface.co/datasets/wikiann),
tokenizes each sentence and compares it.

Unfortunately, some sentences have a slightly different output compared to the `fairseq` tokenizer, but this happens not quite often.
The output of the `xlm_v_tokenizer_comparison.py` script with all tokenizer differences can be viewed [here](tokenizer_diff.txt).

# MLM checks

After the model conversion and tokenizer checks, it is time to check the MLM performance:

```python
from transformers import pipeline

unmasker = pipeline('fill-mask', model='stefan-it/xlm-v-base')
unmasker("Paris is the <mask> of France.")
```

It outputs:

```json
[{'score': 0.9286897778511047,
  'token': 133852,
  'token_str': 'capital',
  'sequence': 'Paris is the capital of France.'},
 {'score': 0.018073994666337967,
  'token': 46562,
  'token_str': 'Capital',
  'sequence': 'Paris is the Capital of France.'},
 {'score': 0.013238662853837013,
  'token': 8696,
  'token_str': 'centre',
  'sequence': 'Paris is the centre of France.'},
 {'score': 0.010450296103954315,
  'token': 550136,
  'token_str': 'heart',
  'sequence': 'Paris is the heart of France.'},
 {'score': 0.005028395913541317,
  'token': 60041,
  'token_str': 'center',
  'sequence': 'Paris is the center of France.'}]
```

Results for masked LM are pretty good!

# Downstream task performance

The last part of integrating a model into 🤗 Transformers is to test the performance on downstream tasks and compare their
performance with the paper results.

For this reason, the `flair-fine-tuner.py` fine-tunes a model on the English WikiANN (Rahimi et al.) split with the hyper-parameters,
mentioned in the paper (only difference is that we use 512 as sequence length compared to 128!). We fine-tune 5 models with
different seeds and average performance over these 5 different models. The scripts expects a model configuration as first input argument.
All configuration files are located under the `./configs` folder. Fine-tuning XLM-V can be started with:

```bash
$ python3 flair-fine-tuner.py ./configs/xlm_v_base.json
```

Fine-tuning is done on a A100 (40GB) instances from [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud) using Flair.
A 40GB is definitely necessary to fine-tune this model with that given batch size! Latest Flair master (commit `23618cd`) is also needed.

## MasakhaNER v1

The script `masakhaner-zero-shot.py` performs zero-shot evaluation on the MasakhaNER v1 datset, that is used in the XLM-V paper.
One crucial part is to deal with `DATE` entities: they do not exist in the English WikiANN (Rahimi et al.) split, but they are
annotated in MasakhaNER v1. For this reason, we convert all `DATE` entities into `O` to disable them for evaluation. The script
`masakhaner-zero-shot.py` is used for performing zero-shot evaluation and will output a nice results table.

Detailed results for all 5 different models can be seen here:

* [XLM-R (Base) Results (Development and Test result)](masakhaner_zero_shot_xlm_r_results.md)
* [XLM-V (Base) Results (Development and Test result)](masakhaner_zero_shot_xlm_v_results.md)

Here's the overall performance table (inspired by Table 11 in the XLM-V paper with their results):

| Model              | amh  | hau  | ibo  | kin  | lug  | luo  | pcm  | swa   | wol  | yor  | Avg.
| ------------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ----
| XLM-R (Paper)      | 25.1 | 43.5 | 11.6 |  9.4 |  9.5 |  8.4 | 36.8 | 48.9  |  5.3 | 10.0 | 20.9
| XLM-R (Reproduced) | 27.1 | 42.4 | 14.2 | 12.4 | 14.3 | 10.0 | 40.6 | 50.2  |  6.3 | 11.5 | 22.9
| XLM-V (Paper)      | 20.6 | 35.9 | 45.9 | 25.0 | 48.7 | 10.4 | 38.2 | 44.0  | 16.7 | 35.8 | 32.1
| XLM-V (Reproduced) | 25.3 | 45.7 | 55.6 | 33.2 | 56.1 | 16.5 | 40.7 | 50.8  | 26.3 | 47.2 | 39.7

Diff. between XLM-V and XLM-R in the paper: (32.1 - 20.9) = 11.2%.

Diff. between reproduced XLM-V and XLM-R: (39.7 - 22.9) = 16.8%.

## WikiANN ([Rahimi et al.](https://aclanthology.org/P19-1015/))

The script `wikiann-zero-shot.py` performs zero-shot evaluation on the WikiANN (Rahimi et al.) dataset. Ths script `wikiann-zero-shot.py`
is used for zero-shot evaluation and will also output a nice results table. Notice: it uses a high batch size for evaluating the model,
so a A100 (40GB) GPU is definitely useful.

Detailed results for all 5 different models can be seen here:

* [XLM-R (Base) Results (Development and Test result)](wikiann_zero_shot_xlm_r_results.md)
* [XLM-V (Base) Results (Development and Test result)](wikiann_zero_shot_xlm_v_results.md)

Here's the overall performance table (inspired by Table 10 in the XLM-V paper with their results):

| Model              |  ro  |  gu  |  pa  |  lt  |  az  |  uk  |  pl  |  qu  |  hu  |  fi  |  et  |  tr  |  kk  |  zh  |  my  |  yo  |  sw
| ------------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
| XLM-R (Paper)      | 73.5 | 62.9 | 53.6 | 72.7 | 61.0 | 72.4 | 77.5 | 60.4 | 75.8 | 74.4 | 71.2 | 75.4 | 42.2 | 25.3 | 48.9 | 33.6 | 66.3
| XLM-R (Reproduced) | 73.8 | 65.5 | 50.6 | 74.3 | 64.0 | 76.5 | 78.4 | 60.8 | 77.7 | 75.9 | 73.0 | 76.4 | 45.2 | 29.8 | 52.3 | 37.6 | 67.0   
| XLM-V (Paper)      | 73.8 | 66.4 | 48.7 | 75.6 | 66.7 | 65.7 | 79.5 | 70.0 | 79.5 | 78.7 | 75.0 | 77.3 | 50.4 | 30.2 | 61.5 | 54.2 | 72.4
| XLM-V (Reproduced) | 77.2 | 65.4 | 53.6 | 74.9 | 66.0 | 69.4 | 79.8 | 66.9 | 79.0 | 77.9 | 76.2 | 76.8 | 48.5 | 28.1 | 58.4 | 62.6 | 71.6 

| Model              |  th  |  ko  |  ka  |  ja  |  ru  |  bg  |  es  |  pt  |  it  |  fr  |  fa  |  ur  |  mr  |  hi  |  bn  |  el  | de
| ------------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
| XLM-R (Paper)      |  5.2 | 49.4 | 65.4 | 21.0 | 63.1 | 76.1 | 70.2 | 77.0 | 76.9 | 76.5 | 44.6 | 51.4 | 61.5 | 67.2 | 69.0 | 73.8 | 74.4
| XLM-R (Reproduced) |  4.7 | 49.4 | 67.5 | 21.9 | 65.2 | 77.5 | 76.7 | 79.0 | 77.7 | 77.9 | 49.0 | 55.1 | 61.3 | 67.8 | 69.6 | 74.1 | 75.4   
| XLM-V (Paper)      |  3.3 | 53.0 | 69.5 | 22.4 | 68.1 | 79.8 | 74.5 | 80.5 | 78.7 | 77.6 | 50.6 | 48.9 | 59.8 | 67.3 | 72.6 | 76.7 | 76.8
| XLM-V (Reproduced) |  2.6 | 51.6 | 71.2 | 20.6 | 67.8 | 79.4 | 76.2 | 79.9 | 79.5 | 77.5 | 51.7 | 51.5 | 61.9 | 69.2 | 73.2 | 75.9 | 77.1

| Model              |  en  |  nl  |  af  |  te  |  ta  |  ml  |  eu  |  tl  |  ms  |  jv  |  id  |  vi  |  he  |  ar  | Avg.
| ------------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
| XLM-R (Paper)      | 83.0 | 80.0 | 75.8 | 49.2 | 56.3 | 61.9 | 57.2 | 69.8 | 68.3 | 59.4 | 48.6 | 67.7 | 53.2 | 43.8 | 61.3
| XLM-R (Reproduced) | 83.4 | 80.8 | 75.8 | 49.3 | 56.8 | 62.2 | 59.1 | 72.2 | 62.3 | 58.3 | 50.0 | 67.9 | 52.6 | 47.8 | 62.6    
| XLM-V (Paper)      | 83.4 | 81.4 | 78.3 | 51.8 | 54.9 | 63.1 | 67.1 | 75.6 | 70.0 | 67.5 | 52.6 | 67.1 | 60.1 | 45.8 | 64.7
| XLM-V (Reproduced) | 84.1 | 81.3 | 78.9 | 50.9 | 55.9 | 63.0 | 65.7 | 75.9 | 70.8 | 64.8 | 53.9 | 69.6 | 61.1 | 47.2 | 65.0

Diff. between XLM-V and XLM-R in the paper: (64.7 - 61.3) = 3.4%.

Diff. between reproduced XLM-V and XLM-R: (65.0 - 62.6) = 2.4%.

# 🤗 Transformers Model Hub

After all checks (weights, tokenizer and downstream tasks) the model can be uploaded to the 🤗 Transformers Model Hub:

* [`stefan-it/xlm-v-base`](https://huggingface.co/stefan-it/xlm-v-base)

Model will be moved to [`Meta AI`](https://huggingface.co/facebook) organization soon.
