# Ordivo v2 data card — draft, no completed corpus

**Status:** planned multi-source research collection; no frozen v2 training mixture or real-data manifest is certified by this card. It documents the intended evidence types and acquisition gates. Actual counts, hashes, licenses/terms, retention and model-release rights remain source-specific. See the [dataset register](V2_DATASET_REGISTER.md) and [release criteria](V2_RELEASE_CRITERIA.md).

## Sources and what their labels mean

| Source | Supervision | Complete unit and split policy | Access/rights status |
|---|---|---|---|
| Ordivo synthetic MCDM | Exact formula rankings, ties and scores conditioned on supplied method/weights | Scenario family assigned before generation; retain all alternatives | Existing v1 control only; v2 corpus pending |
| Expedia Personalized Sort | Observed clicks and bookings, separately | Search group; search-ID split; inspect entity/time overlap | Official file listing succeeded; official data download returned HTTP 403 on 2026-10-19. No source rows acquired; applicable training/derivative rights unresolved |
| Amazon ESCI | Human graded product relevance in a source-sampled judged candidate pool | Query and locale; preserve official test | Official source/build verified at revision and hashes below; repository declares Apache-2.0, retain notices and verify actual artifact rights |
| MSLR-WEB10K | Graded web relevance in a source-sampled judged candidate pool | Official fold and query groups | Agreement acceptance and file records pending |
| Swissmetro | Stated travel-mode choice and availability | Respondent-disjoint panel split | Official source/build verified; source SHA-256 and four-way group counts recorded below. Dataset-specific raw redistribution terms remain unresolved |
| SUSHI (optional) | Elicited rankings | Respondent-disjoint; retain shown alternatives/partial order | Archive/version and terms pending |

Click and booking are noisy behavioral outcomes affected by exposure. Stated choice, relevance and explicit ranking do not measure the same latent quantity. Synthetic solver labels measure a formula, not human utility. No common utility scale or pooled accuracy is implied. Some sources have no authoritative criterion directions or weights; those fields stay absent or `null` until supplied by a declared policy.

## Schema and processing

The shared episode contract is `ordivo-behavioral-v1`: namespaced episode ID and source, task type, group ID, context, criteria/constraints/preferences, complete alternatives with typed features and availability, a task-matching supervision discriminant, exposure, and provenance. A training projection allowlists task/context/feature semantics and declared decision inputs. It excludes supervision, exposure, source IDs, group IDs, row order and provenance. Candidate IDs are keys, remapped to aliases for d1 prompts. Unavailable alternatives remain recorded and masked; a positive outcome on one is a validation error. Missing values remain missing until a transform fitted on training groups only.

For each acquired source, publish a metadata manifest containing URL/revision, raw file SHA-256 and dictionary version, applicable license/terms record, intended use and distribution decision, raw/retained rows and complete groups, feature list/units/availability, label and missingness counts, duplicate/conflict audit, split-policy ID and group-overlap diagnostics, exclusion counts, and remote artifact location. The public card does not contain raw rows, individual search queries or restricted identifiers. Any train-fitted vocabulary, scaler or imputer receives its own hash. Source-specific target/gain policies must be fixed before final evaluation.

## Split and quality requirements

Preserve official test sets where provided. Freeze four disjoint roles: training, model-selection validation, calibration, and final test. Derive the middle two from source training data without touching official tests. Expedia splits whole searches; Swissmetro and SUSHI split respondents; ESCI splits queries including locale; MSLR keeps one published fold. Cross-source duplication and repeated entities require separate diagnostics. Every expected group and candidate in the source-defined set must be accounted for, including singleton, zero-positive, multi-positive, unjudged, unsupported, overflow and failed groups. An unjudged item is not a grade-zero negative. NDCG evaluates ordering within the judged candidate pool; it cannot establish retrieval recall over the full catalog or web. Do not filter individual rows to make a model fit context length. A small fixture validates code paths only; it is not a measured dataset distribution.

The verified first Swissmetro build used official Biogeme source revision
`8d1063b5c371f09f5dd44f430b3be66455f2b785`, source file SHA-256
`27432693cf052985d79a950b4b888be3efca798fc89b0d3ffefe40608ede00f2`.
Of 10,728 raw survey rows, nine had no stated choice and 10,719 complete
three-mode episodes were retained. Respondent-grouped partitions contain 8,532
train, 549 model-selection validation, 594 calibration, and 1,044 final-test
episodes under policy `swissmetro-respondent-v2-calibration`. These counts and
source hash come from the remote source/build receipt; the first classical
held-out results and their separate aggregate/readback are in the
[experiment report](../reports/ordivo-v2-experiment.html). The source describes
the dataset for research and education; raw-data redistribution remains a
separate rights question.

The first official Amazon ESCI Task 1 reduced-ranking build used source
revision `7916cdf6ab75a462e77f20ab40428a10923998d5` and example/product
file SHA-256 values
`4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a` /
`25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265`.
Of 2,621,288 source example rows, 1,503,277 outside the declared Task 1
reduced pool were omitted; 1,118,011 candidate rows formed 48,300
source-defined query groups. Four partitions contain 30,372 train, 1,688
selection, 1,744 calibration, and 14,496 official-test groups. These are
**candidate-pool** counts, not retrieval coverage across the product catalog.
The acquisition/build exited 0 remotely; an explicit cross-partition overlap
and official-test audit remained pending at that build. A separate classical
baseline scored all 14,496 source-defined test query groups under the frozen
old manifest SHA-256
`012fa5f58aaa6b75b0d73c1a9f84f811713e87c61df1dc5137867d34c2cba540`.
Its [aggregate and readback](../reports/ordivo-v2-experiment.html) establish
that run on that manifest only. A later source-bound manifest adds explicit
group accounting; do not conflate the two without exact split-hash comparison.

## Risks and limitations

Observed choices are exposed and position-biased, and Expedia's randomized-presentation flag alone does not provide known examination propensities. Hotel price semantics depend on site/country conventions. ESCI/MSLR relevance is assessor judgment, not purchase behavior. Swissmetro responses are stated rather than revealed preferences. Synthetic numbers inherit generator assumptions and may not represent real tradeoffs. Text coverage differs across locales and candidate set lengths. Incomplete source access or restrictive terms may limit any public adapter even if a private research benchmark can be run.

## Reproduction and distribution

The public release should provide manifests, code/config revisions, split-assignment policy, source links and eligible aggregate statistics. Reproducing restricted source experiments may require the reader's own authorized access and agreement acceptance. The v1 synthetic dataset/card is historical and its published Hub licensing does not license the entire v2 mixture. No source rows, fitted encodings based on restricted IDs, individual prediction tables, or trained derivatives are redistributed until their permissions are documented. Reporting reviewed aggregate scientific measurements is a separate question from redistributing source rows or trained weights. The intended model and its limits are described in the [draft model card](V2_MODEL_CARD_DRAFT.md).

## Expedia rights reading for this draft

Reviewed 2026-10-18 against the [official Kaggle data page](https://www.kaggle.com/competitions/expedia-personalized-sort/data), the [2013 official competition rules PDF](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3504/media/Expedia%20Personalized%20Sort%20Official%20Competition%20Rules%20Final.pdf), and the [current rules tab](https://www.kaggle.com/competitions/expedia-personalized-sort/rules). The public data page labels the license **“Subject to Competition Rules”** and says outside data is prohibited for competition modeling. The PDF says registration supplied data to develop competition entries, prohibits external data for entries, and defines sponsor materials as confidential with purpose-limited use. Its prize-winner license grants rights **to the sponsor**, not a general public license for participants' trained weights. The current rules tab did not expose readable terms to this unauthenticated review. These are source-text observations and a conservative release interpretation, not a conclusion that all private research is forbidden.

| Artifact or action | What the reviewed text establishes | Current release handling |
|---|---|---|
| Authenticated access and private research benchmark | The competition data page offers access subject to rules; the historical PDF centers use on competition entries. It does not explicitly grant Ordivo's later mixed-domain research use. | Continue the user's authorized, access-controlled investigation; retain exact applicable terms and scope with the acquired-source receipt. Do not describe research authorization as a public license. |
| Raw rows, source IDs, per-search examples or prediction tables | The data page supplies no open-data license; PDF confidentiality language limits disclosure of sponsor material. | Keep private. No public row or example excerpt without a documented source-specific permission. |
| Sanitized aggregate metrics and source metadata | Neither page explicitly addresses this category. Aggregates may still reveal restricted information at fine granularity. | Publish only reviewed coarse aggregates after a disclosure check; keep small cells and reconstructable slices private. |
| Source-independent code and synthetic fixtures | The PDF's external-data ban is a competition-entry rule; it does not itself prohibit publishing independently written code. | Publish code/fixtures without Expedia rows, IDs, fitted artifacts or derived private statistics; retain separate project licensing. |
| Expedia-trained adapter or mixed-domain public weights | No clause located in the reviewed public text grants participants permission to distribute trained derivatives. The prize-winner grant runs in the opposite direction, to the sponsor. | Keep any Expedia-trained checkpoint private until applicable current terms or a rights-holder clarification establishes distribution permission. A public v2 path can use only sources with verified derivative-model rights, with Expedia reported as a private benchmark if the applicable terms allow it. |

The concrete unresolved question is whether the account's applicable current competition terms permit **post-competition research training** and **public redistribution of weights derived from this data**. The official CLI could list files but the `DownloadDataFile` request returned HTTP 403 Forbidden on 2026-10-19; that response does not identify the underlying cause, and no file was acquired or terms accepted by automation. Record the authenticated rule/version and any source-specific clarification before assigning an open license to those weights. Do not infer a derivative-model license from file listing, from the 2013 prize clause, or from the user's authorization to use Expedia.
