# ViralMetrics: Understanding What Makes Content Go Viral

**Group 4:** Phoo Pwint Cho Thar (67011755) · Bhawat Kolkitchaiwan (67011087) · Thiri Thaw (67011731) · San Aung (67011725)

> **Research question:** Which measurable content and posting features most strongly predict whether a piece of content goes viral, and can we turn that into honest, testable recommendations?

## Overview

We use features available at the time of publication to predict whether a news article will go **viral** (the top 20% of articles by shares in the training period, or at least 3,500 shares). We compare the models with simple baselines, evaluate them using 5-fold CV, and test them once on a later time period. We only make recommendations when the results remain significant after Bonferroni correction and are also supported by the later test period.

## Dataset

[Online News Popularity (UCI ML Repository, id 332)](https://archive.ics.uci.edu/dataset/332/online+news+popularity): 39,644 Mashable articles, Jan 2013 – Dec 2014, ~58 pre-publication features (day, topic, title/article length, images, videos, links, keywords, sentiment) and the target `shares`.

## How to run

1. Open `ViralMetrics_Group4.ipynb` in Google Colab.
2. **Runtime → Run all** (≈ 15 min on the free tier).

The notebook downloads the dataset from UCI when you run it. If the download doesn't work, put `OnlineNewsPopularity.csv` inside a `data/` folder. Figures are saved in `figures/`, while tables and JSON files are saved in `results/`.

**Requirements:** Python 3, numpy, pandas, scipy, scikit-learn (1.6.1 used), matplotlib, seaborn (all preinstalled on Colab).

## Method

| Step | Detail |
|---|---|
| Cleaning | Removed 1 impossible row, dropped near-constant `n_non_stop_words`, added `no_text` flag |
| Split | Chronological: train 31,637 articles (to 2014-08-27), test 8,006 (from 2014-08-28) |
| Target | `viral` = shares ≥ 80th percentile of training (3,500). Viral rate: 20.5% train, 16.6% test |
| Feature sets | 47 creator-controllable features; 59 with platform-history (`kw_*`, `self_reference_*`) |
| Models | Baselines (majority, stratified) → Logistic Regression → Random Forest / Gradient Boosting |
| Validation | 5-fold stratified CV; test set used once. Main metric: PR-AUC |
| Interpretation | Grouped permutation importance, partial dependence, adjusted odds ratios, error analysis |
| Robustness | Shuffled labels, top-10% target, random split, multiple seeds |

## Key results

**Champion: Random Forest on creator-controllable features** (chosen by CV PR-AUC 0.341 ± 0.002).

| Test period (8,006 articles) | Score | No skill |
|---|---|---|
| PR-AUC | 0.291 | 0.166 |
| ROC-AUC | 0.696 | 0.500 |
| Top-10% scored articles that go viral | 34.1% (2.1× base rate) | 16.6% |

- Logistic regression performs almost as well (test PR-AUC 0.301); adding platform-history features raises PR-AUC to about 0.338.
- Shuffled-label training collapses to near no-skill (PR-AUC 0.185), so the signal is not an artefact.
- Predicting the actual share count is hard: R² ≈ 0.11 on log(shares).
- **Topic has the biggest influence**: the mix of topics and channels ranks highest in permutation importance, while presentation choices have a smaller effect.

## Recommendations

| Area | Recommendation | Train lift | Test lift | Strength |
|---|---|---|---|---|
| Topic | Prioritise social-media / lifestyle over world news | 2.42× | 2.02× | Strong |
| Images | Use image-rich layouts (6+ images); a single image does worst | 1.71× | 2.34× | Strong |
| Timing | Publish flagship pieces on Saturday / Sunday | 1.44× | 1.49× | Strong |
| Links | Link generously to sources and related content | 1.38× | 1.25× | Strong |
| Videos | Embed video when the story supports it | 1.34× | 1.26× | Strong |
| Article length | Don't cut long-form pieces short | 1.27× | 1.07× | Weak |
| Title length | No recommendation (no real effect) | 1.01× | 1.07× | Fails |

These factors are associated with a higher *chance* of going viral, but none of them guarantees success. Run an A/B test before committing resources.

## Limitations

- The dataset covers Mashable articles from 2013–2014 only. It does not include posting hour, video length, or audio information.
- The data is observational, so the results show associations rather than cause and effect.
- At the time the data was collected, test articles had been online for less time than training articles (median of 72 days compared with 427 days). This may partly explain their lower viral rate.
- `kw_*` history features may leak later information, so recommendations use only the controllable model.
- The model can help rank articles, but it cannot predict virality with certainty. About 3 out of 4 articles it flags as likely to go viral do not.

## Repository contents

```
ViralMetrics_Group4.ipynb   # full analysis (sections 2–10)
README.md
data/                       # created on first run (UCI download)
figures/                    # saved plots
results/                    # cv_results.csv, test_results.csv, adjusted_odds_ratios.csv,
                            # recommendations.csv, results.json
```

## Reference

Fernandes, K., Vinagre, P., Cortez, P. (2015). *A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News.* EPIA 2015.
