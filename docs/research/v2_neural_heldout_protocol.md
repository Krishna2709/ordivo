# Bounded paired neural held-out probe

This is an exploratory follow-up to a completed supervised pilot, not a fresh
protected release test. Swissmetro classical test metrics were already visible
when this probe was specified. Freeze the committed
`configs/v2_neural_heldout.yaml` and exact pilot config/receipt before running;
do not change training, feature policies or selection after viewing results.
That config is frozen for the Swissmetro+ESCI fallback pilot. The separate
`configs/v2_neural_heldout_expedia_esci.yaml` applies only if an Expedia+ESCI
pilot completes; the runner requires exact source-key agreement.

The remote runner verifies the pilot's completed receipt, saved best adapter,
base model revision, exact feature policy, four split hashes and group-disjoint
lineage for each source. It selects up to 50 complete test split units per
source by SHA-256 of seed, source and unit identity, independent of labels.
Every row of a selected respondent panel stays together. It renders the same
complete candidate prompt once and uses that frozen prompt/readout for base d1
and the best validation-selected adapter. Candidate identity stays outside the
prompt; unavailable options are masked. A prompt that exceeds the pilot's
token limit is counted as overflow, never truncated.

For stated choice and exactly-one-booking exposed groups, the paired metrics
are accuracy@1, conditional negative log-likelihood, and multiclass Brier.
Booking groups with zero or multiple bookings and groups with fewer than two
available options are excluded with counts before model inference. ESCI
reports NDCG@10 under the source E/C/S/I gain policy; zero-ideal-gain groups
are scored but excluded from that metric with a count. The runner rejects any
missing candidate or model prediction and writes an incomplete failure ledger
instead of publishing a reduced successful aggregate. Per-query probabilities
and metrics stay under remote `/workspace`; the aggregate contains only source
and split metadata, denominators and paired differences. No temperature is
fitted or calibration claim made.

The ESCI classical baseline currently uses title and description lexical
features; the fallback neural pilot uses title, brand and color. Any
comparison between them has different feature budgets. A matched-input
control and a fresh protected release protocol are required before claiming
neural superiority or general-purpose transfer.

## S0b three-domain exploratory reuse, not yet run

`configs/v2_neural_heldout_s0b.yaml` freezes seed 42, a 600-second task cap,
and at most 50 whole test split units each for Swissmetro, ESCI, and Expedia.
It reuses the same paired evaluator after a valid S0b pilot receipt exists.
The evaluator loads fresh pinned base d1 and a separate fresh base with the
S0b best adapter, then scores the exact same rendered candidate prompts. It
enforces S0b's pilot feature policies and prompt limit; Expedia's
`conditional_booking_choice` admits only exactly-one-booking searches, with
zero/multiple booking and overflow exclusions counted explicitly. All
candidate scores and selected episodes must be covered or the run fails.

Swissmetro and ESCI source manifests, feature policies, and seed are proposed
to remain the same as the first exploratory probe. Before comparing, check
that S0b's selected-unit hashes equal the first probe's Swissmetro
`c546c7af44ce27b4cfb8988d6621222cbd230881fd68f32bfa9b2356843f97dc`
and ESCI `f4af35cfddaa544cdac993014d4e131e80814b2daeb0e713579fa2132181d15a`.
The earlier ESCI probe admitted 27 of 50 selected groups because 23 prompts
exceeded its shorter token limit. Report S0b's full new admitted cohort and
the episode-ID intersection with those original 27 separately, with paired
base/S0b metrics on the intersection and any missing old IDs disclosed. A
longer S0b prompt limit may expand coverage, but the new aggregate must not
silently substitute that larger population for the old one. The original
per-query files are retained privately under
`/workspace/ordivo-v2-data/neural-heldout/paired-854de-retry/`.

This test cohort has already been exposed to exploratory analysis. Neither
S0b nor its hyperparameters may be selected or retuned using these results.
The probe is not a protected release test or a claim that S0b's changes caused
any observed difference. Schedule it before the separate post-S0b validation
candidate-order diagnostic if remote GPU time permits; do not access the test
split for that validation-only diagnostic.
