#!/usr/bin/env python3
"""
certificate_audit.py — Audit geo_cert certificates for consistency and monotonicity.

Three audit checks:

1. S_ceiling monotonicity: S_ceiling should increase (or hold) as model versions
   improve. A model with higher R² should leave less room for S. If S_ceiling
   goes *up* between versions for the same task, the newer model is explaining
   less structure — either a regression or the task is noisy.

2. N_ceiling invariance: N_ceiling is a per-task constant (Definition A).
   Within bootstrap CI, all models should agree on the same N_ceiling[task].
   If a model's implied N deviates beyond CI, the ceiling ensemble is
   inconsistent for that task.

3. Flagged-task handling: Tasks flagged by the N_ceiling estimator (high
   residual correlation) get a warning annotation on their certificates
   rather than suppression — the certificate is still valid, but consumers
   should treat the N_ceiling bound as optimistic.

Usage:
    from apps.geo_cert.certificates.audit import CertificateAuditor

    auditor = CertificateAuditor(ceiling_report, cert_rows)
    findings = auditor.audit()
    auditor.print_findings(findings)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from apps.geo_cert.certificates.issuer import GeoCertRow
from apps.geo_cert.models.ceiling.n_ceiling_estimator import (
    CeilingEstimateReport,
    TaskCeilingResult,
)

log = logging.getLogger(__name__)


@dataclass
class AuditFinding:
    """Single audit finding."""
    check: str          # "s_monotonicity" | "n_invariance" | "flagged_task"
    severity: str       # "error" | "warning" | "info"
    task: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AuditReport:
    """Complete audit report."""
    findings: list          # list[AuditFinding]
    n_errors: int
    n_warnings: int
    n_info: int
    n_tasks_audited: int
    n_certs_audited: int
    passed: bool            # True if n_errors == 0


class CertificateAuditor:
    """Audit geo_cert certificates against the N_ceiling estimate.

    Args:
        ceiling_report: CeilingEstimateReport from NCeilingEstimator.
        cert_rows: list of GeoCertRow from GeoCertIssuer.
    """

    def __init__(
        self,
        ceiling_report: CeilingEstimateReport,
        cert_rows: list,
    ):
        self.ceiling = ceiling_report
        self.rows = cert_rows

        # Index ceiling results by task
        self._ceiling_by_task = {
            r.task: r for r in ceiling_report.tasks
        }

    def audit(self) -> AuditReport:
        """Run all audit checks.

        Returns:
            AuditReport with all findings.
        """
        findings = []
        findings.extend(self._check_s_monotonicity())
        findings.extend(self._check_n_invariance())
        findings.extend(self._check_flagged_tasks())
        findings.extend(self._check_simplex_integrity())

        n_errors = sum(1 for f in findings if f.severity == "error")
        n_warnings = sum(1 for f in findings if f.severity == "warning")
        n_info = sum(1 for f in findings if f.severity == "info")
        tasks = set(r.task for r in self.rows)

        report = AuditReport(
            findings=findings,
            n_errors=n_errors,
            n_warnings=n_warnings,
            n_info=n_info,
            n_tasks_audited=len(tasks),
            n_certs_audited=len(self.rows),
            passed=n_errors == 0,
        )

        log.info(
            f"Audit complete: {n_errors} errors, {n_warnings} warnings, "
            f"{n_info} info across {len(tasks)} tasks, {len(self.rows)} certs"
        )
        return report

    # ------------------------------------------------------------------
    # Check 1: S_ceiling monotonicity across model versions
    # ------------------------------------------------------------------

    def _check_s_monotonicity(self) -> list:
        """Check that S_ceiling decreases (or holds) as R² improves.

        For each task, sort models by R². S_ceiling = 1 - R - N where N is
        constant per task. So S should decrease as R increases. If S goes
        up between a worse and better model, something is wrong.

        Since N_ceiling is fixed per task, S_ceiling = 1 - R_ceiling - N_ceiling.
        Monotonicity of S w.r.t. R is algebraically guaranteed when N is constant.
        This check catches cases where the issuer computed inconsistent values.
        """
        findings = []

        # Group rows by (task, model_version), take first row per group
        # (R_ceiling and S_ceiling are per-model, not per-ZCTA)
        task_models = {}
        for r in self.rows:
            key = (r.task, r.model_version)
            if key not in task_models:
                task_models[key] = r

        # Group by task
        by_task = {}
        for (task, mv), row in task_models.items():
            by_task.setdefault(task, []).append((mv, row))

        for task, model_rows in by_task.items():
            # Sort by R_ceiling ascending
            model_rows.sort(key=lambda x: x[1].R_ceiling)

            for i in range(1, len(model_rows)):
                prev_mv, prev = model_rows[i - 1]
                curr_mv, curr = model_rows[i]

                # S should decrease or stay the same as R increases
                if curr.S_ceiling > prev.S_ceiling + 1e-6:
                    findings.append(AuditFinding(
                        check="s_monotonicity",
                        severity="error",
                        task=task,
                        message=(
                            f"S_ceiling increased from {prev.S_ceiling:.4f} "
                            f"({prev_mv}) to {curr.S_ceiling:.4f} ({curr_mv}) "
                            f"despite R_ceiling increasing from "
                            f"{prev.R_ceiling:.4f} to {curr.R_ceiling:.4f}"
                        ),
                        details={
                            "prev_model": prev_mv,
                            "curr_model": curr_mv,
                            "prev_S": prev.S_ceiling,
                            "curr_S": curr.S_ceiling,
                            "prev_R": prev.R_ceiling,
                            "curr_R": curr.R_ceiling,
                        },
                    ))

            # Info: report the S range for this task
            s_values = [r.S_ceiling for _, r in model_rows]
            findings.append(AuditFinding(
                check="s_monotonicity",
                severity="info",
                task=task,
                message=(
                    f"S_ceiling range: [{min(s_values):.4f}, {max(s_values):.4f}] "
                    f"across {len(model_rows)} models"
                ),
                details={
                    "s_min": min(s_values),
                    "s_max": max(s_values),
                    "n_models": len(model_rows),
                },
            ))

        return findings

    # ------------------------------------------------------------------
    # Check 2: N_ceiling invariance within bootstrap CI
    # ------------------------------------------------------------------

    def _check_n_invariance(self) -> list:
        """Check that N_ceiling values used in certificates match the
        ceiling estimate within bootstrap CI.

        N_ceiling is a per-task constant. Every certificate for a given task
        should use the same N_ceiling value, and that value should fall within
        the bootstrap confidence interval from the ceiling estimator.
        """
        findings = []

        # Collect unique N_ceiling values per task from certificates
        task_n_values = {}
        for r in self.rows:
            task_n_values.setdefault(r.task, set()).add(round(r.N_ceiling, 6))

        for task, n_values in task_n_values.items():
            # All certs for a task should use the same N_ceiling
            if len(n_values) > 1:
                findings.append(AuditFinding(
                    check="n_invariance",
                    severity="error",
                    task=task,
                    message=(
                        f"Multiple N_ceiling values in certificates: "
                        f"{sorted(n_values)}"
                    ),
                    details={"n_values": sorted(n_values)},
                ))
                continue

            cert_n = list(n_values)[0]

            # Check against ceiling estimate CI
            ceiling = self._ceiling_by_task.get(task)
            if ceiling is None:
                findings.append(AuditFinding(
                    check="n_invariance",
                    severity="warning",
                    task=task,
                    message=(
                        f"No ceiling estimate for task (N_ceiling={cert_n:.4f} "
                        f"used in certs but cannot validate against CI)"
                    ),
                    details={"cert_n": cert_n},
                ))
                continue

            # Value should match point estimate
            if abs(cert_n - ceiling.n_ceiling) > 1e-4:
                findings.append(AuditFinding(
                    check="n_invariance",
                    severity="error",
                    task=task,
                    message=(
                        f"Certificate N_ceiling={cert_n:.6f} differs from "
                        f"ceiling estimate N_ceiling={ceiling.n_ceiling:.6f}"
                    ),
                    details={
                        "cert_n": cert_n,
                        "estimate_n": ceiling.n_ceiling,
                        "diff": abs(cert_n - ceiling.n_ceiling),
                    },
                ))
            else:
                # Check within bootstrap CI
                in_ci = ceiling.ci_lower <= cert_n <= ceiling.ci_upper
                findings.append(AuditFinding(
                    check="n_invariance",
                    severity="info" if in_ci else "warning",
                    task=task,
                    message=(
                        f"N_ceiling={cert_n:.4f} "
                        f"{'within' if in_ci else 'outside'} "
                        f"bootstrap CI [{ceiling.ci_lower:.4f}, "
                        f"{ceiling.ci_upper:.4f}]"
                    ),
                    details={
                        "cert_n": cert_n,
                        "ci_lower": ceiling.ci_lower,
                        "ci_upper": ceiling.ci_upper,
                        "in_ci": in_ci,
                    },
                ))

        return findings

    # ------------------------------------------------------------------
    # Check 3: Flagged tasks from ceiling estimator
    # ------------------------------------------------------------------

    def _check_flagged_tasks(self) -> list:
        """Annotate certificates for tasks flagged by the ceiling estimator.

        Flagged tasks have high mean residual correlation across models,
        meaning the ceiling models are capturing similar structure and
        N_ceiling may be optimistic. Certificates are still valid but
        consumers should treat the bound cautiously.
        """
        findings = []

        for ceiling_result in self.ceiling.tasks:
            if not ceiling_result.flagged:
                continue

            n_certs = sum(
                1 for r in self.rows if r.task == ceiling_result.task
            )

            findings.append(AuditFinding(
                check="flagged_task",
                severity="warning",
                task=ceiling_result.task,
                message=(
                    f"Task flagged for high residual correlation "
                    f"(mean |rho|={ceiling_result.mean_residual_correlation:.3f}). "
                    f"N_ceiling={ceiling_result.n_ceiling:.4f} may be optimistic. "
                    f"{n_certs} certificates affected."
                ),
                details={
                    "mean_residual_correlation": ceiling_result.mean_residual_correlation,
                    "n_ceiling": ceiling_result.n_ceiling,
                    "n_certs": n_certs,
                    "best_model": ceiling_result.best_model,
                    "residual_correlation_matrix": ceiling_result.residual_correlation_matrix,
                },
            ))

        return findings

    # ------------------------------------------------------------------
    # Check 4: Simplex integrity on every certificate
    # ------------------------------------------------------------------

    def _check_simplex_integrity(self) -> list:
        """Verify R + S + N = 1 on every certificate (ceiling and proxy).

        Uses YRSNCertificate.verify() where available, plus direct checks
        on the GeoCertRow decomposition values.
        """
        findings = []
        violations = 0

        for r in self.rows:
            # Ceiling decomposition
            total_c = r.R_ceiling + r.S_ceiling + r.N_ceiling
            if abs(total_c - 1.0) > 1e-4:
                violations += 1
                if violations <= 5:  # Cap detailed findings
                    findings.append(AuditFinding(
                        check="simplex_integrity",
                        severity="error",
                        task=r.task,
                        message=(
                            f"Ceiling simplex violation: "
                            f"R={r.R_ceiling:.6f} + S={r.S_ceiling:.6f} + "
                            f"N={r.N_ceiling:.6f} = {total_c:.6f} != 1.0 "
                            f"(zcta={r.zcta}, model={r.model_version})"
                        ),
                        details={
                            "zcta": r.zcta,
                            "model": r.model_version,
                            "R": r.R_ceiling,
                            "S": r.S_ceiling,
                            "N": r.N_ceiling,
                            "total": total_c,
                        },
                    ))

            # Proxy decomposition
            total_p = r.R_proxy + r.S_proxy + r.N_proxy
            if abs(total_p - 1.0) > 1e-4:
                violations += 1
                if violations <= 5:
                    findings.append(AuditFinding(
                        check="simplex_integrity",
                        severity="error",
                        task=r.task,
                        message=(
                            f"Proxy simplex violation: "
                            f"R={r.R_proxy:.6f} + S={r.S_proxy:.6f} + "
                            f"N={r.N_proxy:.6f} = {total_p:.6f} != 1.0 "
                            f"(zcta={r.zcta}, model={r.model_version})"
                        ),
                        details={
                            "zcta": r.zcta,
                            "model": r.model_version,
                            "R": r.R_proxy,
                            "S": r.S_proxy,
                            "N": r.N_proxy,
                            "total": total_p,
                        },
                    ))

            # YRSNCertificate.verify() on both certs
            for label, cert in [("ceiling", r.cert_ceiling), ("proxy", r.cert_proxy)]:
                v = cert.verify()
                if not v.get("simplex_valid", True):
                    violations += 1
                    if violations <= 5:
                        findings.append(AuditFinding(
                            check="simplex_integrity",
                            severity="error",
                            task=r.task,
                            message=(
                                f"YRSNCertificate.verify() failed simplex "
                                f"check on {label} cert "
                                f"(zcta={r.zcta}, model={r.model_version})"
                            ),
                            details={"label": label, "verify": v},
                        ))

        if violations > 5:
            findings.append(AuditFinding(
                check="simplex_integrity",
                severity="error",
                task="(aggregate)",
                message=f"{violations} total simplex violations (showing first 5)",
                details={"total_violations": violations},
            ))
        elif violations == 0:
            findings.append(AuditFinding(
                check="simplex_integrity",
                severity="info",
                task="(all)",
                message=f"All {len(self.rows)} certificates pass simplex integrity",
                details={"n_checked": len(self.rows)},
            ))

        return findings

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_findings(report: AuditReport) -> None:
        """Print audit findings to stdout."""
        print("\n" + "=" * 80)
        print("CERTIFICATE AUDIT REPORT")
        print("=" * 80)
        print(f"Tasks: {report.n_tasks_audited}, "
              f"Certificates: {report.n_certs_audited}")
        print(f"Result: {'PASSED' if report.passed else 'FAILED'}")
        print(f"  Errors:   {report.n_errors}")
        print(f"  Warnings: {report.n_warnings}")
        print(f"  Info:     {report.n_info}")
        print()

        for severity in ("error", "warning", "info"):
            items = [f for f in report.findings if f.severity == severity]
            if not items:
                continue
            label = severity.upper()
            print(f"--- {label} ({len(items)}) ---")
            for f in items:
                print(f"  [{f.check}] {f.task}: {f.message}")
            print()

        print("=" * 80)
