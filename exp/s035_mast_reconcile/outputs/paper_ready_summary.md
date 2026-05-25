# Paper-ready S035 Summary

S035-MAST-Reconcile validates the methodological lineage behind GeoCert. Prior work in `stress_mast.py` operationalized MAST as signal-pattern stress tests: taxonomy specifications, synthetic failure injection, diagnostic scoring, and a stress suite. GeoCert reuses this operational pattern, but replaces MAST's multi-agent execution labels with a native evaluation-failure taxonomy over label construction, decomposition reduction, and deployment translation.

The rerun produced a standalone GeoCert stress harness with nine failure modes. Synthetic injection validation achieved top-1 accuracy of 0.3333 and top-3 accuracy of 0.7778 across the nine design-time failure signatures. Applied to S018D summary evidence, the auditor identifies gate compression, range compression, proxy calibration drift, and noisy-control/tercile-uniformity pathologies as the dominant failure candidates.

The caveat is important: the sandbox run proves artifact-level lineage through hashes and zip metadata, not git-level provenance. It also diagnoses S018D from summary metrics rather than raw per-sample ablation tensors. The repo-side rerun should add commit hashes and raw-output reclassification before final submission.
