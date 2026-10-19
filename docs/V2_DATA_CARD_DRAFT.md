# Ordivo v2 data card — draft, private research corpora

**Status:** Swissmetro and ESCI source-bound manifests and a preliminary two-domain training subset have remote receipts; their processed four-way splits and notices are retained with SHA readback in private dataset repo `blackOrder/Ordivo-v2-research-data` at commit `92df99f88ab12dd9c016f62e08ea15206600e98c`. Expedia official source has a complete grouped four-way build and bounded 5,000-search classical ranking baseline; its full and sampled processed splits passed private Hub file-level readback at commit `58d0b26d558278ab49f4c0769b44ed928d44204c`. There is no Expedia neural training. SUSHI has a separate evaluation-only four-way source control, with no record upload or neural training. There is no frozen **public-release** v2 training mixture. Counts and hashes below refer to specific executed builds; source rights and model-release rights remain source-specific. The [private research-data README](../hub/v2_research_data/README.md) describes the controlled Hub layout. See the [dataset register](V2_DATASET_REGISTER.md) and [release criteria](V2_RELEASE_CRITERIA.md).

## Sources and what their labels mean

| Source | Supervision | Complete unit and split policy | Access/rights status |
|---|---|---|---|
| Ordivo synthetic MCDM | Exact formula rankings, ties and scores conditioned on supplied method/weights | Scenario family assigned before generation; retain all alternatives | Existing v1 control only; v2 corpus pending |
| Expedia Personalized Sort | Observed clicks and bookings, separately | Search group; search-ID split; inspect entity/time overlap | Earlier official requests returned HTTP 403. After user-confirmed rules, official archive/CSV and grouped four-way build were acquired/hashed; a bounded sampled classical benchmark but no neural training. Public trained-derivative rights remain unresolved |
| Amazon ESCI | Human graded product relevance in a source-sampled judged candidate pool | Query and locale; preserve official test | Official source/build verified at revision and hashes below; repository declares Apache-2.0, retain notices and verify actual artifact rights |
| MSLR-WEB10K | Graded web relevance in a source-sampled judged candidate pool | Official fold and query groups | Agreement acceptance and file records pending |
| Swissmetro | Stated travel-mode choice and availability | Respondent-disjoint panel split | Official source/build verified; source SHA-256 and four-way group counts recorded below. Dataset-specific raw redistribution terms remain unresolved |
| SUSHI (evaluation-only control) | Elicited complete order of the ten exposed items | Respondent-disjoint; 10,000 episodes/100,000 candidates; roles 7,912/518/586/984 | [Official SUSHI terms](https://www.kamishima.net/sushi/) permit research use and prohibit redistribution. Corrected archive and [four-way split audit](../reports/runs/v2-sushi-20261019/fourway-audit.json) verified (manifest SHA-256 `bd6a19be2715ee3c43b45607b4c0177bd318c124f4c17dab082a14d3f866b6fa`). No records uploaded, neural training or derivative-weight rights claim. |

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
The acquisition/build exited 0 remotely. A subsequent
[source-bound audit](../reports/runs/v2-source-audit-20261019/audit.json)
reconciled all four old/new ESCI split SHA-256 values exactly, found zero
query-and-locale overlap across roles, and confirmed all 14,496 official-test
groups remain in test. The same audit reconciled Swissmetro's four split hashes.
A separate classical
baseline scored all 14,496 source-defined test query groups under the frozen
old manifest SHA-256
`012fa5f58aaa6b75b0d73c1a9f84f811713e87c61df1dc5137867d34c2cba540`.
Its [aggregate and readback](../reports/ordivo-v2-experiment.html) establish
the run on that manifest. The new source-bound manifest adds explicit group
accounting while retaining byte-identical split files; the audit does not
constitute a new model evaluation or uncertainty estimate.

The official Expedia exposed-candidate build used `train.csv` SHA-256
`84a53139ea92df1e9400f84852facbd78f3136f366a08440f4ab4518e71a49a9`
at executed data commit `919c67e494174f34d2436fa507e427b75fe866c2`.
It retained 9,917,530 of 9,917,530 rows and all 399,344 search groups. Four
search-disjoint partitions contain 319,714 train, 19,739 selection, 20,040
calibration and 39,851 final-test groups. The
[sanitized manifest](../reports/runs/v2-expedia-20261019/manifest.json)
records split SHA-256 hashes and policy
`expedia-2013-split-v2-calibration`, exposed candidate scope, and explicit
click/booking judgment policy. There are 122,751 groups with no booking and
25,969 groups with multiple clicks; neither should be silently converted into
a single-booking choice label. Missing `prop_location_score2` and
`prop_review_score` counts are 2,178,380 and 14,630 candidate rows. The source build itself contains no Expedia ranking result, trained adapter, or public row redistribution right; the bounded ranking run below is separate.
An independent [four-way streaming audit](../reports/runs/v2-expedia-20261019/fourway-audit.json)
(SHA-256 `d48db0af0be72b48c7244ca41e78dda5789c496bf6a29fc7a8bcb5211558c387`)
re-read all split hashes and accounted for 399,344 unique search groups and
9,917,530 candidates, with zero duplicate groups/candidate IDs, source
mismatches or wrong-hash partitions. This is integrity evidence, not
prediction quality.

A later Expedia classical benchmark used a label-independent SHA-256 sample
of **complete** source search groups, capped at 20,000 train, 2,000 selection,
2,000 unused calibration and 5,000 test. Its
[bounded manifest](../reports/runs/v2-expedia-bounded-20261019/manifest.json)
SHA-256 is `c72552b8fe1acd0cab0480610dd65b5dd6aed575313047d2da2f2786953b4a2b`;
the sampled test contains 123,456 candidates and excludes 34,851 of the
original source test searches (868,472 candidates). All 5,000 selected test
searches were scored with zero failures. The
[sanitized aggregate](../reports/runs/v2-expedia-bounded-20261019/aggregate.json)
SHA-256 `95d9251e7efa1d9b6ab745f25cd8649fb9bbf5d532bdcaa76dab6d39db1182b0`
reports declared click/booking 0/1/5-gain NDCG@10: LambdaRank 0.413110,
price 0.268414, rating 0.240019, seeded random 0.232711 on 5,000 exposed
candidate groups. Booking-only NDCG@10 0.436156 is defined on 3,389 groups
with positive booking gain; 1,611 zero-booking groups are excluded from that
endpoint. This is a bounded ranking benchmark, not full-source performance,
booking probability or an Expedia-trained neural result.

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

The concrete unresolved question is whether the account's applicable current competition terms permit **post-competition research training** and **public redistribution of weights derived from this data**. An earlier official `DownloadDataFile` request returned HTTP 403 Forbidden; it did not identify the underlying cause. After the user confirmed acceptance of the Kaggle rules, an official archive was acquired and hashed, with no terms accepted by automation. The [path-redacted inventory](../reports/runs/v2-expedia-20261019/inventory.json) records 434,573,097 bytes and SHA-256 `f16b9622a73c015e2693284a5939a82817417f23f284bfd69081f1f35cdd0602`. Its CSV members use Deflate64 compression; initial Python extraction failed, then 7-Zip 23.01 extracted the official `train.csv`. The [sanitized acquisition receipt](../reports/runs/v2-expedia-20261019/acquisition.json) records 2,358,114,793 bytes and SHA-256 `84a53139ea92df1e9400f84852facbd78f3136f366a08440f4ab4518e71a49a9`. A later private remote build at `919c67e494174f34d2436fa507e427b75fe866c2` retained all 9,917,530 exposed candidate rows in 399,344 complete search groups. The [sanitized grouped manifest](../reports/runs/v2-expedia-20261019/manifest.json) (SHA-256 `3b14e3f30466a2da868ac636c8888bc42366f865737715a57a67f32866a02d92`) records 319,714 train, 19,739 selection, 20,040 calibration and 39,851 final-test groups, with no omitted rows or groups. This is a grouped source corpus, **not** a measured Expedia baseline or neural result. Record the applicable accepted rule/version and any source-specific clarification before assigning an open license to trained weights. Do not infer a derivative-model license from download success, from the 2013 prize clause, or from the user's authorization to use Expedia.
