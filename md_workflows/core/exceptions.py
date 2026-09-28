"""Structured exceptions raised by core functions.

Core code never calls ``sys.exit`` or terminates the process — it raises these so any
caller (CLI, SDK, or a future orchestrator) can catch and handle them. The CLI layer
is the only place that maps these to process exit codes.
"""

from __future__ import annotations

from pathlib import Path


class MDWorkflowError(Exception):
    """Base class for all md-workflows domain errors."""


class MissingInputError(MDWorkflowError):
    """A step was asked to run but one or more required input files are absent.

    Raised by the input guard (``check_inputs``) before any external tool is invoked.
    """

    def __init__(self, step: str, missing: list[Path]):
        self.step = step
        self.missing = list(missing)
        listed = "\n  ".join(str(p) for p in self.missing)
        super().__init__(f"{step}: missing required input file(s):\n  {listed}")


class StepToolError(MDWorkflowError):
    """An external tool (gmx, tleap, ChimeraX, AmberTools, ...) exited non-zero."""

    def __init__(
        self,
        tool: str,
        returncode: int,
        *,
        cmd: list[str] | None = None,
        stderr: str | None = None,
        log_path: Path | None = None,
    ):
        self.tool = tool
        self.returncode = returncode
        self.cmd = list(cmd) if cmd else None
        self.stderr = stderr
        self.log_path = log_path
        parts = [f"{tool} exited with code {returncode}"]
        if cmd:
            parts.append("command: " + " ".join(str(c) for c in cmd))
        if log_path:
            parts.append(f"log: {log_path}")
        if stderr:
            tail = "\n".join(stderr.strip().splitlines()[-20:])
            parts.append(f"stderr (tail):\n{tail}")
        super().__init__("\n".join(parts))


class CrystalSymmetryError(MDWorkflowError):
    """A PDB lacks the symmetry metadata needed to build the crystal, or the requested
    crystal cannot be written as a PDB file.

    Raised by the gemmi primitives in ``core.crystal`` — missing/unknown CRYST1 space
    group, a bad symmetry-operation order, or a supercell whose numbering would not fit
    the fixed-column serial/resSeq fields.
    """


class GmxOutputParseError(MDWorkflowError):
    """A value expected in GROMACS stdout/log (e.g. a molecule count) was not found.

    These parses are tied to gmx output strings that shift across versions, so failures
    are surfaced explicitly rather than as a bare ``RuntimeError``.
    """

    def __init__(self, what: str, source: Path | str, *, gmx_version: str | None = None):
        self.what = what
        self.source = source
        self.gmx_version = gmx_version
        msg = f"could not parse {what} from {source}"
        if gmx_version:
            msg += f" (gmx {gmx_version})"
        super().__init__(msg)
