# Music Genre Classification (GTZAN)

Classifying 30-second music clips into 10 genres from hand-crafted audio features, using
[librosa](https://librosa.org/) for signal analysis and a small Keras neural network for
classification. Built as an end-to-end walkthrough: audio exploration → feature engineering →
EDA → model → evaluation.

**Historical notebook result:** 88.7 % test accuracy on a row-wise split of the 3-second-clip feature set. This is **not a reliable estimate for unseen tracks**; see the evaluation limitations below.

## Dataset

[GTZAN](http://marsyas.info/downloads/datasets.html) — 1 000 tracks, 10 genres
(blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock), 100 tracks each,
30 s per track at 22 050 Hz.

This repository ships the two tabular feature files that the model actually trains on:

| file | rows | description |
|---|---|---|
| `Data/features_30_sec.csv` | 1 000 | 57 features per full 30-second track |
| `Data/features_3_sec.csv` | 9 990 | the same features per 3-second window (10× more training examples) |

The raw `.wav` files are **not** included (≈1.2 GB). They are only needed for the audio-exploration
section of the notebook; download them from the link above (or
[Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification))
and place them under `Data/genres_original/<genre>/`.

## Features

Each row contains the mean and variance of the following per-frame descriptors, plus tempo:

- Chroma STFT, RMS energy, spectral centroid, spectral bandwidth, spectral roll-off, zero-crossing rate
- Harmonic and percussive components (HPSS)
- 20 MFCC coefficients

The notebook visualises how these are computed for individual tracks: waveforms, STFT and mel
spectrograms, zero crossings, spectral centroid/roll-off overlays, MFCC and chroma maps.

![Tempo (BPM) distribution per genre](docs/bpm_boxplot.jpg)

*Tempo alone already separates some genres (e.g. disco vs. classical), but most overlap heavily — which is why the full feature set is needed.*

## Evaluation limitations — read before comparing accuracy

The notebook fits `StandardScaler` before its train/test split, and the split is unseeded and row-wise. Multiple 3-second windows from the same source track can therefore appear on both sides, and preprocessing sees held-out statistics. The reported 88.7% is a historical notebook output, not a newly reproduced or leakage-controlled benchmark.

For a credible follow-up, group windows by their original track, split those groups before preprocessing, fit the scaler on training data only, and report the seed, per-genre metrics and track-disjoint evaluation. GTZAN also has known dataset-quality limitations; a high clip score does not establish production performance. No new accuracy claim is made here.

## Model

```
StandardScaler → Dense(256, relu) → Dense(128, relu) → Dense(64, relu) → Dense(10, softmax)
optimizer=adam · loss=sparse_categorical_crossentropy · 500 epochs · batch 64
```

Train/test split: 60 / 40 on `features_3_sec.csv`. A 2-component PCA of the min-max-scaled
features is also plotted to show the (limited) linear separability of the genres.

| metric | value |
|---|---|
| test accuracy | **0.887** |
| test loss | 1.09 |
| train accuracy (epoch 500) | ≈ 0.999 |

The gap between train and test accuracy indicates over-fitting; see *Ideas* below.

## Start without downloading audio

Use the bundled feature CSVs to inspect the tabular pipeline first. The opening audio-exploration cells need separate WAV files and may contain paths from the author’s environment; adjust paths before running them. Install the notebook requirements in an isolated environment. TensorFlow support depends on your Python version and platform; this repository does not provide a locked, cross-platform environment. Dataset availability does not imply permission to redistribute the original recordings.

## Running it

```bash
git clone https://github.com/RANJITHROSAN17/musicgenre.git
cd musicgenre
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook music_genre_classification.ipynb
```

The feature-based EDA and the model train from the bundled CSVs without any audio download
(runtime depends on hardware and dependency versions; no timing guarantee). The first ~60 cells (audio exploration) require the
GTZAN `.wav` files as described above.

## Project structure

```
music_genre_classification.ipynb   exploration, EDA, model training and evaluation
Data/features_30_sec.csv           per-track features (1 000 rows)
Data/features_3_sec.csv            per-3-second-window features (9 990 rows)
docs/bpm_boxplot.jpg               tempo-per-genre plot used above
requirements.txt
```

## Ideas / next steps

- Regularisation (dropout, early stopping, weight decay) to close the train/test gap
- Stratified, seeded split and k-fold cross-validation for a more reliable estimate
- Compare against gradient-boosted trees and an SVM on the same features
- A CNN on mel-spectrogram images (`images_original/` in GTZAN) instead of tabular features
- Confusion matrix to see which genres are confused (rock vs. country is the classic one)

## Maintainer

**[Ranjith Rosan](https://github.com/RANJITHROSAN17)** maintains this GTZAN music-genre classification walkthrough. GTZAN and third-party libraries retain their original attribution; see the dataset acknowledgements and evaluation limitations above.

## Acknowledgements

GTZAN dataset by G. Tzanetakis and P. Cook, *Musical genre classification of audio signals*,
IEEE Transactions on Audio and Speech Processing, 2002. Feature CSVs follow the format of the
Kaggle distribution of the dataset.
