# Ordivo v2 multi-domain dataset research register

Research date: 2026-10-18 local time. This register implements the user's direction
to use Expedia and a wide variety of data for a general-purpose decision model.
It records candidate sources and adapter decisions, not completed ingestion or
measured model results. No raw datasets were downloaded in this review.

## Initial suite

| Dataset | Evidence type | Episode and leakage-safe split | Role/status |
|---|---|---|---|
| Expedia Personalized Sort / ICDM 2013 | Observed hotel click/booking behavior | Complete search; split search IDs; inspect repeated entities/time separately | Primary numerical-choice adapter; user selected; Kaggle authentication not found locally |
| Amazon ESCI Shopping Queries | Human-annotated product relevance | Query and locale with candidate products; preserve official test and derive validation from train | Primary text/attribute ranking adapter; official public repository identified |
| Microsoft MSLR-WEB10K | Human relevance grades over engineered features | Query groups in a fixed official fold | Independent ranking benchmark; agreement applies |
| Swissmetro | Stated transport-mode choices | One choice situation with available modes; keep respondent panel in one split | Tradeoff and availability benchmark; source located, data-specific terms/fields still to inspect |
| SUSHI | Elicited food-preference rankings | Respondent and shown alternatives; respondent-disjoint splits | Optional explicit-ranking control; not the main evidence for broad applicability |
| Ordivo synthetic MCDM | Deterministic solver supervision | Existing family-based train/val/test | Preserved pretraining and mathematical control; not real behavior |

## Verified source facts and adapter decisions

### Expedia

The [official dictionary](https://www.kaggle.com/c/expedia-personalized-sort/data)
defines search/property IDs, click/booking outcomes, and random displayed ordering.
Its test file omits outcomes. Use held-out groups from labeled data for research
evaluation. The displayed price convention varies; customer-to-property distance
does not mean city-center distance. Keep outcomes and presentation metadata out
of the serving projection. Do not fabricate explicit preference weights.

User approval covers using Expedia. Retrieve through an authenticated remote
account when configured. The dictionary and [historical rules](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3504/media/Expedia%20Personalized%20Sort%20Official%20Competition%20Rules%20Final.pdf)
do not establish unrestricted redistribution or pretrained-model release rights;
record applicable terms with the acquired source. Do not silently substitute a
third-party mirror or accept new contractual obligations on the user's behalf.

### Amazon ESCI

The [official Amazon repository](https://github.com/amazon-science/esci-data)
provides query-product judgments in English, Japanese, and Spanish, with product
text/attributes and E/S/C/I categories. Candidate lists can reach 40 products.
It publishes reduced ranking and larger classification variants and query-based
train/test splits. Start with the reduced ranking variant and keep locale in
join keys; join products on `(product_id, product_locale)` and preserve every
expected candidate. Unknown joins fail or are explicitly recorded, never silently
removed. Labels express relevance, not purchase probability or personal utility.

The repository declares [Apache-2.0](https://github.com/amazon-science/esci-data/blob/main/LICENSE).
Pin the repository commit and actual file metadata at ingestion. Preserve
license/notices and inspect any file-specific terms. Large LFS files stay remote.
Do not assume model tokenizer limits support complete 40-item text-rich groups.
Use a lexical baseline plus a declared text-feature ranker before neural training.

### MSLR-WEB10K

The [Microsoft release](https://www.microsoft.com/en-us/research/project/mslr/)
provides grouped relevance ranking data and requires acceptance of an agreement.
Preserve a fixed published fold; parse query groups and grades without converting
them into behavior labels. Engineered features make this a useful numerical
ranking control but do not prove reasoning over named real-world criteria.
The exact feature inventory, file hashes, grade gains and agreement record are
ingestion deliverables, not assumed completed work.

### Swissmetro

The [official Biogeme example](https://biogeme.epfl.ch/sphinx/auto_examples/swissmetro/plot_b01a_logit.html)
uses stated preferences over train, Swissmetro, and car, with travel-time,
cost, and availability terms. The [source loader](https://biogeme.epfl.ch/sphinx/_modules/biogeme/data/swissmetro.html)
exposes respondent ID, choice and availability fields. Treat these as stated,
not observed market choices. Split repeated observations by respondent; preserve
availability masks and report discarded/no-choice cases. A multinomial-logit
model is a necessary specialist baseline. Dataset provenance and terms must be
recorded separately from the Biogeme software license.

### SUSHI

The [author's dataset page](https://www.kamishima.net/sushi/) provides elicited
preference data and distinguishes ranking from scoring collections, including
a warning about older scoring files. Pin a corrected archive and its README
if used. Preserve original presented sets and partial-order semantics; do not
manufacture negatives for unseen items. This small preference benchmark adds
a supervision type but is insufficient as the principal generality evidence.

## Shared-model requirements

- One context/candidate input boundary with typed numeric, categorical, text,
  missingness and availability features; task-specific supervision remains outside.
- No universal gain scale or universal winner conversion across datasets.
- Domain-specific baselines and a shared model are both necessary. Several
  separately trained models do not demonstrate general-purpose transfer.
- Uniform domain/query sampling is the initial engineering default; report
  repeats and cap small-domain exposure. Validate weighting on source domains.
- Leave an entire domain out of training and tuning for each transfer fold.
  Test within-domain, unseen-domain, and declared few-shot adaptation separately.
- Query-, respondent-, and official-fold split policies are different and
  versioned. Report cross-source duplication and language/domain coverage.
- Scores, latency, constraint compliance and uncertainty are reported by task
  and domain. No single pooled accuracy headline.

## Acquisition record required per source

Source URL and revision; download/file SHA-256; access/terms reference; intended
use and artifact permissions; dictionary version; raw and retained rows/groups;
feature availability/provenance; label semantics; missingness/duplicate report;
split policy and grouping entity; target/gain policy; candidate-set integrity;
and remote storage location. Public documentation can contain these metadata
while raw rows remain in the permitted remote storage scope.

Expedia authentication is the immediate access dependency. ESCI research and
the source-independent adapter framework can proceed without it. Actual runs
require a remote runtime; this document does not start paid compute.
