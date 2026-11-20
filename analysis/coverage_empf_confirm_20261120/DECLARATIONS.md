# Declarations locked before the first official confirmation score

Companion to PROTOCOL.md, which is hash-pinned by score.py's source manifest
(pinned at the version of commit a662ad3) and therefore not edited further.


Accepted by the study lead after the scoring build, review and repair
(workflow wf_441e966e-e85), before `score.py` wrote any score for a
confirmation fit:

- Readout A for the paired B-minus-A comparison is regenerated on CPU through
  the pilot's own path, so that A and B share common random numbers as in the
  pilot (the CLI writes A on CUDA, which is not common-random with a CPU B).
  The CLI's CUDA archive is scored too, as `A_cli`, and reported as secondary.
- Question 1 and question 2 decide on the protocol's literal rule (coverage of
  the mean over fits). The mean of per-fit |coverage - 0.8| is printed as a
  diagnostic only; it cannot change a decision.
- Secondary references, folds 0-2: the stream-matched covariance-challenge
  files used for the owner's EMPF selection
  (`analysis/covariance_challenge_20261114/artifact_audit_complete_20261116.json`);
  folds 3-8: the count-shape confirmation and `resample_blind` files pinned in
  `default_config_distribution_evidence_20261113.json`. Paired SEs are
  descriptive, because seeds share generation streams and folds 3-8 share one
  empirical stream per fold.
- Disclosure: while checking the code, the reviewer computed readout A and B
  for one confirmation fit (fold 4, seed 0) with its own scripts. Those
  numbers are not official scores and were not used to set any gate,
  threshold or read-out; none of the gates or thresholds above changed after
  that review. The official fold 4 seed 0 score is produced by `score.py`
  like every other fit.
