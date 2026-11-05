---
license: cc-by-sa-4.0
language:
- tr
library_name: transformers
pipeline_tag: token-classification
tags:
- pii
- personal-data
- privacy
- kvkk
- turkish
- onnx
base_model:
- dbmdz/electra-small-turkish-cased-discriminator
- dbmdz/bert-base-turkish-cased
datasets:
- turkish-nlp-suite/turkish-wikiNER
- turkish-nlp-suite/MusteriYorumlari
- BTX24/turkish-privacy-pii-ner
- negentropi/belgin-pii-dataset
- openai/gsm8k
---

# akana-pii-tagger

Two token classifiers that find personal data in Turkish text. They are the learned PII tagger of
[akana](https://github.com/altaidevorg/akana), a Turkish NLP toolkit in Rust, which downloads them
from this repository on first use and runs their ONNX exports on the CPU.

| Folder | Model | Used in | ONNX size |
|---|---|---|---:|
| `electra-small/` | ELECTRA-small, fine-tuned from `dbmdz/electra-small-turkish-cased-discriminator` | fast mode | 56.6 MB |
| `berturk/` | BERTurk, fine-tuned from `dbmdz/bert-base-turkish-cased` | accurate mode, averaged with the small model on every window | 441.9 MB |

Each folder holds `model.onnx` (what akana runs), `model.safetensors`, `config.json` and the
tokenizer.

## Labels

Nine labels with B- and I- tags, 19 tags in all (`config.json`'s `id2label`):

| Label | What it marks |
|---|---|
| PERSON | names of people: full names, given names or surnames alone, nicknames used as names, in any case or spelling; public figures included |
| USERNAME | social-media handles and account or login names |
| ADDRESS | a postal address precise enough to find a home or workplace (a city or district alone is not one) |
| PHONE | phone and fax numbers |
| EMAIL | e-mail addresses |
| URL | personal web addresses only: a person's profile, links carrying a private token or record id |
| ID | national, financial, customer, transaction, vehicle, device and court-case numbers |
| SECRET | passwords, PINs, one-time codes, CVV, API keys, access tokens, private keys |
| DATE | personal dates only: birth, appointment, purchase, delivery, payment, a private person's hearing; not holidays, laws, public events or publication dates |

Special categories of personal data (health, religion and the like) are not tagged.

## How akana runs them

1. The text is cut into windows of 254 WordPiece tokens (256 with `[CLS]` and `[SEP]`); neighbouring
   windows share 64 tokens, and each token takes its scores from the window where it sits farthest
   from an edge.
2. Fast mode uses the small model's tag log-probabilities. Accurate mode averages the probabilities
   of both models on every window (about five times slower).
3. A BIO-constrained Viterbi pass over the whole text gives the spans, widened to whole words.
4. A policy leaves call-centre and service numbers (444, 0850, 0800, 0900, short codes) and role
   mailboxes (`info@`, `destek@`, `kvkk@`, ...) unmasked.

The **strict** operating points add a bias to the PERSON, USERNAME and DATE tags: +5.25 in fast
mode and +2.75 in accurate mode. Each was chosen by conformal risk control (Angelopoulos et al.,
2022) on 632 development texts that contain personal data, so that the expected share of a text's
personal characters left in clear is at most 4 % (α = 0.04). The bound holds on average over texts
like the calibration texts, not for every text.

## Use

With akana's command-line tool (built with the `pii-tagger` feature), which downloads the models
into its cache on first use, pinned to a commit of this repository and checked by SHA-256:

```bash
akana pii-tag "Ayşe Yılmaz 12 Mart 2024'te 0532 123 45 67 numaralı telefondan aradı."
akana pii-tag --accurate --strict --file mektup.txt
akana pii-models --accurate   # download both models now
```

Spans are byte offsets with their labels. `AKANA_CACHE_DIR` sets the cache, `AKANA_OFFLINE=1`
turns downloads off, and `HF_ENDPOINT` points them at a mirror.

With `transformers`, for per-token tags (without akana's windowing, Viterbi decoding and policy,
so results differ somewhat from akana's):

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

repo = "altaidevorg/akana-pii-tagger"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="electra-small")
model = AutoModelForTokenClassification.from_pretrained(repo, subfolder="electra-small")
tag = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="first")
print(tag("Ayşe Yılmaz 12 Mart 2024'te 0532 123 45 67 numaralı telefondan aradı."))
```

## Evaluation

Scored once on held-out sets, after the models and their operating points were fixed on
development data. A value is found **exactly** when a predicted span is the gold span, whatever
its label (edge punctuation, apostrophe suffixes and titles aside), and **fully covered** when every
character of it is masked; **leak** is the share of personal characters left in clear; false
alarms are counted per 10,000 characters of texts that hold no personal data.

**Real Turkish text.** The test split of akana's real-text gold: 1,617 units from court decisions,
Wikipedia, news, tweets, product reviews, voice-assistant requests, chatbot prompts, social-media
comments and maths word problems, holding 900 personal values (791 person names, 57 dates, 47
identifiers, a few phones, e-mails and secrets), and 1,149 units that hold none. The gold comes from
at least two independent labellings by Claude models, adjudicated value by value and corrected on
review.

| Mode | Precision | Recall | Exact F1 (95% CI) | Fully covered | Leak | False alarms /10k |
|---|---:|---:|---:|---:|---:|---:|
| fast | 0.868 | 0.884 | 0.876 (0.853–0.897) | 0.913 | 7.0% | 4.3 |
| accurate | 0.935 | 0.917 | 0.926 (0.908–0.943) | 0.934 | 5.3% | 1.3 |
| fast, strict | 0.687 | 0.916 | 0.785 (0.747–0.818) | 0.963 | 2.4% | 12.0 |
| accurate, strict | 0.812 | 0.937 | 0.870 (0.843–0.893) | 0.966 | 2.6% | 6.6 |

In both strict modes a held-out text with personal data had, on average, 3.8% of its personal
characters left in clear (468 texts; 95% interval about 2.2–5.4%), against the 4% the calibration
aims at.

**Other held-out sets**, exact F1 in fast / accurate mode:

- [yusuf-said/turkish-privacy-filter-dataset](https://huggingface.co/datasets/yusuf-said/turkish-privacy-filter-dataset),
  a fixed 70% test part (3,537 synthetic support, log, form and OCR texts, 6,670 values): 0.988 /
  0.995. Of its 898 texts without personal data, 5.8% / 2.3% get a span.
- Turkish WikiNER test sentences, person names only (1,000 sentences, 654 names): 0.892 / 0.918.
- [Saturday Labs' masking benchmark](https://huggingface.co/datasets/cagrigungor/turkish-pii-masking-benchmark),
  scored on the values this tagger's labels cover (550 texts, 932 values): 0.896 / 0.929.
- Belgin and BTX24 test splits: 1.000 / 0.997 and 1.000 / 1.000. Both share templates with their
  training splits, so these figures only show that the templates were learnt.

**Release gates** set by akana's PII audit for a learned detector, on the same held-out sets:

| Gate | Target | fast | accurate | fast / accurate strict |
|---|---|---|---|---|
| Phones, e-mails and account numbers (yusuf-said) | fully covered ≥ 0.99, precision ≥ 0.97 | met | met | met / met |
| Person names in real text | fully covered ≥ 0.95, precision ≥ 0.90 | 0.919, 0.877 | 0.941, 0.944 | 0.970, 0.681 / 0.970, 0.811 |
| Addresses and personal dates (yusuf-said; real text) | fully covered ≥ 0.90 | real-text dates 0.895 | real-text dates 0.895 | met / met (dates 0.982) |
| False alarms on real text | ≤ 1 per 10k characters | 4.3 | 1.3 | 12.0 / 6.6 |
| Texts without personal data flagged (yusuf-said) | ≤ 5% | 5.8% | met (2.3%) | 13.3% / 5.9% |
| Behavioural probes (sentence frames per capability) | ≥ 95% per capability | chat names in lower case with a suffix 83%, public look-alike numbers 94% | met | not run |

The date gate is missed by one date: the fast and accurate modes fully mask 51 of the 57 personal
dates in real text, where 52 are needed. Five of the six dates they miss are in court decisions.

On four threads of a laptop CPU (Intel i5-12500H), akana tags 2 KB of text in about 60–90 ms in
fast mode and 0.8 s in accurate mode.

## Training data

Fine-tuned for token classification on synthetic texts, two public Turkish PII datasets with their
values refilled, WikiNER's person names, generated hard cases and 15,065 units of real Turkish
text (court decisions, news, tweets, voice-assistant requests, chatbot prompts, social-media
comments, Wikipedia, translated maths word problems and product reviews) labelled by Claude Sonnet
under akana's annotation guideline. `DATA_SOURCES.md` lists every source with its licence;
`wikipedia_attribution.tsv` lists the Wikipedia articles and revisions the text was taken from.
The evaluation sets above were never used to train, tune or choose the models: real-text training
units come from other documents than the evaluation units and share no passage with them, and
BTX24, Belgin and WikiNER were trained on their train splits and scored on their test splits.

## Limitations

- Turkish only. Informal text is the weakest: on development data, social-media comments score
  0.68 exact F1 in accurate mode, against 0.91 over all domains.
- Personal dates: the fast and accurate modes leave about one in ten personal dates of real text in
  clear, mostly in court decisions (see the date gate above); the strict modes left 1 of 57.
- Training labels come from a language model and from generated data, so their mistakes carry
  over. Whether a date or URL is personal is a judgement the labels do not always make the same way
  (an order date in "X tarihli ürün geldi", for instance).
- Public codes and numbers (of notices, regulations, products) are sometimes tagged as IDs, and
  names inside organisation or street names are sometimes tagged as persons.
- No mode finds everything: use it as one layer of a privacy process, with review where a leak
  matters.

## Licence

CC BY-SA 4.0 (`LICENSE`): the weights learnt from text under CC BY-SA 4.0 (Wikipedia, WikiNER,
MusteriYorumlari). Reuse must credit the sources listed in `DATA_SOURCES.md` and share adaptations
under the same licence. The base models are MIT-licensed.
