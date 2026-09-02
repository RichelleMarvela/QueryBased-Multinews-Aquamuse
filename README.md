# AQuaMUSE — Query-Based Multi-Document News Summarization

AQuaMUSE is an experimental pipeline for query-based multi-document abstractive news summarization. This repository contains an end-to-end Jupyter notebook and supporting code to prepare data (web scraping + cleaning), perform EDA, extract keywords, train a BART summarization model, and evaluate using ROUGE, METEOR and BERTScore.

Project notebook: `AQuaMUSE_Summarization_Project_FIXED (1).ipynb`

---

## Highlights / Quick summary

- Pipeline: Dataset Prep → EDA → Preprocessing → Text Representation → Keyword Extraction → Model Training → Evaluation
- Experiments: 1 model (BART-large-cnn) × 4 keyword input types (no_query, baseline, RAKE, KBmini) × 2 hyperparameter configs → 8 experiments
- Keyword extraction methods used:
  - RAKE (trained/extracted during pipeline)
  - KeyBERT (MiniLM) — used as a semantic keyword guide
  - YAKE & KeyBERT-Dist used only for EDA/analysis (not trained)
- Dataset: Hugging Face `google-research-datasets/aquamuse` (abstractive split)

---

## Repository structure (not exhaustive)

- AQuaMUSE_Summarization_Project_FIXED (1).ipynb  — main experiment notebook
- `scraping_cache.json` — caching for scraped pages (generated)
- `df_train_scraped.csv`, `df_val_scraped.csv`, `df_test_scraped.csv` — scraped raw documents (generated)
- `df_train_clean.csv`, `df_val_clean.csv`, `df_test_clean.csv` — cleaned dataset (generated)
- README.md — this file

---

## Key design decisions

- Use hybrid scraping (trafilatura primary, BeautifulSoup fallback) to robustly extract article text from URLs.
- Cache scraped pages keyed by MD5 hash of URL list to avoid repeated network calls.
- Apply strict cleaning: remove documents <50 or >5000 words, remove noisy/404-like pages, deduplicate by query.
- Since BART-large-cnn has input length limits (~1024 tokens), the notebook includes sentence-level selection/truncation strategies to fit long multi-doc inputs.
- Evaluate using ROUGE (1,2,L), METEOR, and BERTScore for complementary perspectives.

---

## Requirements

The notebook includes pip install commands for the dependencies; core packages used:

- Python 3.8+
- torch (CUDA recommended for training)
- transformers
- datasets
- accelerate
- sentence-transformers
- keybert, rake-nltk, yake
- trafilatura, beautifulsoup4, requests
- scikit-learn, gensim, nltk
- rouge-score, bert-score
- pandas, numpy, matplotlib, seaborn, wordcloud, pyLDAvis, umap-learn
- jupyter / ipywidgets

Install example:
```bash
pip install datasets transformers torch torchvision torchaudio sentencepiece accelerate
pip install rake-nltk yake keybert sentence-transformers rouge-score bert-score
pip install scikit-learn gensim requests beautifulsoup4 trafilatura lxml nltk
pip install pandas numpy tqdm matplotlib seaborn wordcloud pyLDAvis umap-learn jupyter ipywidgets
```

Note: GPU (CUDA) is highly recommended for model fine-tuning.

---

## How to run (quickstart)

1. Clone the repo:
   git clone https://github.com/RichelleMarvela/QueryBased-Multinews-Aquamuse.git

2. Create and activate your Python environment, install dependencies (see above).

3. Open the notebook:
   jupyter notebook "AQuaMUSE_Summarization_Project_FIXED (1).ipynb"

4. Run cells sequentially. Key long-running steps:
   - Data scraping: populates `df_*_scraped.csv` and uses `scraping_cache.json`.
   - Data cleaning: produces `df_*_clean.csv`.
   - Keyword extraction (RAKE, KeyBERT) and EDA plots.
   - Model training (BART-large-cnn) via Hugging Face Transformers Trainer/Seq2SeqTrainer.

---

## Data details

- Original dataset loaded from HF: `google-research-datasets/aquamuse` (abstractive).
- The notebook fetches article contents from `input_urls` (max 3 URLs per example).
- Scraped texts are concatenated using a ` [SEP_DOC] ` delimiter to form multi-document inputs.
- Caching avoids repeated scraping across runs.

---

## Preprocessing & input strategy

- Documents are cleaned, tokenized with NLTK utilities, and filtered by length.
- For long inputs (over model limit), use sentence scoring / selection or truncate while prioritizing sentences most relevant to the query/keywords.
- Keyword-guided inputs tested:
  - no_query: document only
  - baseline: query + full document
  - RAKE: query + RAKE-extracted keywords
  - KBmini: query + KeyBERT (paraphrase-MiniLM-L6-v2) keywords

---

## Model & Training

- Model: `facebook/bart-large-cnn` (fine-tuned for abstractive summarization).
- Trainer: Hugging Face `Seq2SeqTrainer` (notebook shows example training arguments).
- Two hyperparameter configurations (`cfg1`, `cfg2`) were used in experiments to study sensitivity.

Notes:
- Use mixed precision (FP16) when using CUDA: speeds up training and reduces memory.
- Enable gradient checkpointing to reduce memory footprint for large models.

---

## Evaluation

- Metrics: ROUGE-1, ROUGE-2, ROUGE-L, METEOR, BERTScore.
- The notebook includes functions to compute and aggregate metrics on validation/test splits and to produce per-example comparisons.

---

## Results / EDA (summary)

(Refer to the notebook for full visualizations and numeric results.)

- Average document length after scraping: ~972 words (most between 500–1500).
- Query lengths are typically short (≤ 20 words).
- Unigram overlap (document vs. target summary) average ≈ 0.57 — dataset exhibits a mixed extractive/abstractive behavior.
- Validation/test splits were reduced during cleaning to ensure quality gold-standard references (note: validation/test loss due to strict filters is intentional for quality).

---

## Reproducibility & tips

- Random seed: notebook uses `RANDOM_SEED = 42`.
- Scraping cache: keep `scraping_cache.json` to avoid repeated scraping and to make runs efficient/reproducible.
- Save `df_*_scraped.csv` and `df_*_clean.csv` outputs to share experiment data with collaborators.
- Consider using a smaller BART variant for quick experiments, then scale to `bart-large-cnn` for final runs.

---

## Troubleshooting

- If scraping fails frequently, check `scraping_cache.json` and your network/CORS; the notebook uses a polite `User-Agent` header and small delays between requests.
- If GPU memory errors occur: lower batch size, enable gradient checkpointing, use FP16, or use a smaller model.

---

## Contributing

If you'd like to contribute:
- Open an issue describing the change/bug you'd like to address.
- Fork the repo, create a branch, add tests or a reproducible example, and submit a pull request.

---

## License

This repository is provided for academic/research use. Add or replace with your preferred license (e.g., MIT, Apache 2.0) as required.

---

## Contact / Acknowledgements

Author: Richelle Marvela  
Repo: https://github.com/RichelleMarvela/QueryBased-Multinews-Aquamuse

Based on the AQuaMUSE dataset from Google Research. Uses Hugging Face Transformers and many community libraries (trafilatura, KeyBERT, RAKE, etc.).
