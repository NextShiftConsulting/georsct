# S035 Git/Artifact Proof

This sandbox rerun was executed from uploaded artifacts, not from a live git checkout. Therefore true commit hashes are not available in this environment. The proof available here is artifact-level proof: zip entry timestamps, file sizes, CRCs, and SHA-256 hashes.

## Current proof

- `mast_study.zip` contains `stress_mast.py` and `__init__.py`.
- `stress_mast.py` implements the MAST/YRSN pattern: taxonomy specs, failure injection, diagnosis, and a stress suite.
- `stress_geocert.py` implements the same operational pattern for GeoCert's evaluation-failure taxonomy.

## Required repo-side proof to add

When this is rerun in the source repository, append:

```bash
git log --follow -- experiments/.../stress_mast.py
git log --follow -- experiments/.../stress_geocert.py
git rev-parse HEAD
git diff --stat
```

## Artifact hashes

- `mast_study.zip`: `ef53ff19ccf52c7782e5ef633484cc78865a5572dec4729ec6a8fb4c91dde6c7` (8109 bytes)
- `mast70_Why_Do_Multiagent_Systems_F.pdf`: `dcffd1e4e761cee1fe6ec9c405d3656258276835b67f28c717a392c647b93f5f` (1762078 bytes)
- `Intelligence_as_Representation_Solver_Compatibility.pdf`: `fb0b36d035dad193d7f06b2248118704d640e75d3044ef0cf5ce88bf0879c641` (394857 bytes)
- `Pasted text(163).txt`: `10e45950630bb10503b24d3ac3a37cca62ed48ca0c6b510a1c9c24001195bdb6` (7641 bytes)

## Zip entries

- `stress_mast.py`: timestamp=2026-02-25 00:47:58, size=28334, crc=b76bdd11
- `__init__.py`: timestamp=2026-02-25 00:47:58, size=1811, crc=5c63ffaf
