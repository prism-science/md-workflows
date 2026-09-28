"""Build an equilibrated bulk-water reservoir matching the crystal unit cell.

Corresponds to the canonical (taylor) ``make_waterbox.sh``: subdivide the cell by
``nc_scale``, fill the sub-cell with water at ``conc`` mol/L, tile it back up
``nc_scale``x, restore the full crystal CRYST1, then minimize + equilibrate the box.

The tiling is done with gemmi (``core.crystal.propagate_cell``), which replaces the
AmberTools ``PropPDB`` call the shell recipe shelled out to; ``gmx insert-molecules`` is
the only external tool this step still needs. ``nc_scale``^3 cells of water overrun the
fixed-column PDB resSeq field easily (PropPDB wrapped it silently), so the default
``auto`` numbering restarts the counters per cell once the totals no longer fit.

The water count written to the topology is the exact molecule count of the expanded box
(``count_waters``), superseding the scripts' log-derived estimates.
"""

from __future__ import annotations

from pathlib import Path

from ..config import GromacsRunProfile, SystemConfig, WaterboxParams
from ..gmx import checksums, count_waters, run_min, run_tool
from ..paths import under
from ..results import StepInputs, StepResult, StepStatus

STEP = "make_waterbox"


class MakeWaterboxInputs(StepInputs):
    xtal: Path
    wat_pdb: Path
    prot_top: Path
    min_mdp: Path
    equil_mdp: Path

    def consumed_paths(self) -> list[Path]:
        return [self.xtal, self.wat_pdb, self.prot_top, self.min_mdp, self.equil_mdp]


class MakeWaterboxResult(StepResult):
    @property
    def water_equil_gro(self) -> Path:
        return self.output("water_equil_gro")


def resolve_inputs(workdir: Path, cfg: SystemConfig) -> MakeWaterboxInputs:
    workdir = Path(workdir)
    mdp_dir = under(workdir, cfg.mdp_dir)
    return MakeWaterboxInputs(
        workdir=workdir,
        xtal=workdir / "xtal.pdb",
        wat_pdb=workdir / "WAT.pdb",
        prot_top=workdir / "prot.top",
        min_mdp=mdp_dir / cfg.waterbox.min_mdp,
        equil_mdp=mdp_dir / cfg.waterbox.equil_mdp,
    )


def check_inputs(inputs: MakeWaterboxInputs) -> None:
    inputs.check_exists(STEP)


def make_waterbox(
    inputs: MakeWaterboxInputs,
    params: WaterboxParams,
    min_profile: GromacsRunProfile,
    equil_profile: GromacsRunProfile,
    *,
    resume: bool = False,
) -> MakeWaterboxResult:
    # local import keeps module import (and the contract tests) light
    from ..crystal import propagate_cell

    check_inputs(inputs)
    wd = inputs.workdir
    wb = wd / "waterbox"
    wb.mkdir(parents=True, exist_ok=True)
    nc = params.nc_scale

    box_solv_expand = wb / "box_solv_expand.pdb"
    waterbox_top = wb / "waterbox.top"
    water_min_gro = wb / "water_min.gro"
    water_equil_gro = wb / "water_equil.gro"
    outputs = {
        "box_solv_expand": box_solv_expand,
        "waterbox_top": waterbox_top,
        "water_min_gro": water_min_gro,
        "water_equil_gro": water_equil_gro,
    }
    consumed = {
        "xtal": inputs.xtal,
        "wat_pdb": inputs.wat_pdb,
        "prot_top": inputs.prot_top,
        "min_mdp": inputs.min_mdp,
        "equil_mdp": inputs.equil_mdp,
    }

    if resume and water_equil_gro.exists():
        return MakeWaterboxResult(
            step=STEP,
            status=StepStatus.SKIPPED,
            workdir=wd,
            outputs=outputs,
            params=params.model_dump(),
            input_checksums=checksums(consumed),
        )

    cryst1_xtal = _extract_cryst1(inputs.xtal)
    (wb / "cryst1_xtal.pdb").write_text(cryst1_xtal)
    _create_box_pdb(inputs.xtal, wb / "box.pdb", nc)

    # Fill the sub-cell with water, then tile it up nc x nc x nc.
    run_tool(
        [
            "gmx",
            "insert-molecules",
            "-f",
            str(wb / "box.pdb"),
            "-ci",
            str(inputs.wat_pdb),
            "-conc",
            str(params.conc),
            "-o",
            str(wb / "box_solv.pdb"),
        ],
        tool="gmx insert-molecules",
        cwd=wb,
        log_path=wb / "insert-molecules.log",
    )
    propagated = propagate_cell(
        wb / "box_solv.pdb",
        box_solv_expand,
        ix=nc,
        iy=nc,
        iz=nc,
        numbering=params.numbering,
    )
    _restore_cryst1(box_solv_expand, cryst1_xtal)

    nwat = count_waters(box_solv_expand)
    _write_topology(inputs.prot_top, waterbox_top, nwat)

    # Water box has no restraint reference, so equilibration is a plain grompp+mdrun.
    min_run = run_min(box_solv_expand, waterbox_top, inputs.min_mdp, "water_min", min_profile, wb)
    equil_run = run_min(
        min_run.gro, waterbox_top, inputs.equil_mdp, "water_equil", equil_profile, wb
    )

    return MakeWaterboxResult(
        step=STEP,
        status=StepStatus.COMPLETED,
        workdir=wd,
        outputs=outputs,
        params=params.model_dump(),
        input_checksums=checksums(consumed),
        run_profiles_used={"waterbox_min": min_profile, "waterbox_equil": equil_profile},
        metrics={"nwat": nwat, "nc_scale": nc, "box_atoms": propagated.natoms},
        log_paths=[min_run.log, equil_run.log],
    )


def _extract_cryst1(xtal: Path) -> str:
    with open(xtal) as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                return line
    raise ValueError(f"{xtal}: no CRYST1 record found")


def _create_box_pdb(xtal: Path, box_pdb: Path, nc: int) -> None:
    """Write a CRYST1-only box.pdb with cell dimensions divided by ``nc``."""
    with open(xtal) as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                a = float(line[6:15]) / nc
                b = float(line[15:24]) / nc
                c = float(line[24:33]) / nc
                alpha = float(line[33:40])
                beta = float(line[40:47])
                gamma = float(line[47:54])
                box_pdb.write_text(
                    f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f}\n"
                )
                return
    raise ValueError(f"{xtal}: no CRYST1 record found")


def _restore_cryst1(expanded_pdb: Path, cryst1: str) -> None:
    with open(expanded_pdb) as fh:
        body = [line for line in fh if not line.startswith(("CRYST1", "HEADER"))]
    with open(expanded_pdb, "w") as fh:
        fh.write(cryst1)
        fh.writelines(body)


def _write_topology(prot_top: Path, waterbox_top: Path, nwat: int) -> None:
    header: list[str] = []
    with open(prot_top) as fh:
        for line in fh:
            if "molecules" in line.lower():
                break
            header.append(line)
    with open(waterbox_top, "w") as fh:
        fh.writelines(header)
        fh.write("[ molecules ]\n")
        fh.write("; Compound       #mols\n")
        fh.write(f"WAT              {nwat}\n")
