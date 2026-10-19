# Bounded paired neural held-out probe

This is an exploratory follow-up to a completed supervised pilot, not a fresh
protected release test. Swissmetro classical test metrics were already visible
when this probe was specified. Freeze the committed
`configs/v2_neural_heldout.yaml` and exact pilot config/receipt before running;
do not change training, feature policies or selection after viewing results.

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
