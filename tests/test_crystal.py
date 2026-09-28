from pathlib import Path

import gemmi
import pytest

from md_workflows.core.crystal import expand_to_unit_cell, propagate_cell
from md_workflows.core.exceptions import CrystalSymmetryError

CELL = (20.0, 30.0, 40.0, 90.0, 90.0, 90.0)


def _cryst1(spacegroup: str) -> str:
    a, b, c, alpha, beta, gamma = CELL
    return f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} {spacegroup:<11}\n"


def _atom(serial: int, name: str, resname: str, resseq: int, xyz, chain: str = "A") -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {name:<4} {resname:>3} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
    )


def _write_asu(tmp_path: Path, spacegroup: str, name: str = "asu.pdb") -> Path:
    """Three one-atom residues inside the cell, plus a TER record."""
    pdb = tmp_path / name
    pdb.write_text(
        _cryst1(spacegroup)
        + _atom(1, "N", "ALA", 1, (1.0, 2.0, 3.0))
        + _atom(2, "CA", "GLY", 2, (4.0, 5.0, 6.0))
        + _atom(3, "CB", "SER", 3, (7.0, 8.0, 9.0))
        + "TER       3      SER A   3\n"
        + "END\n"
    )
    return pdb


def _coords(pdb: Path) -> list[tuple[float, float, float]]:
    return [
        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        for line in pdb.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]


def _fields(pdb: Path, start: int, stop: int) -> list[str]:
    return [
        line[start:stop]
        for line in pdb.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]


def test_expand_p1_writes_p1_cryst1_and_keeps_coordinates(tmp_path: Path):
    asu = _write_asu(tmp_path, "P 1")
    out = tmp_path / "cell.pdb"
    result = expand_to_unit_cell(asu, out)

    assert result.spacegroup == "P 1"
    assert result.chain_ids == ["A"]
    assert result.natoms == 3
    assert out.read_text().splitlines()[0].endswith("P 1")
    assert _coords(out) == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]


def test_expand_labels_one_chain_per_symmetry_copy(tmp_path: Path):
    asu = _write_asu(tmp_path, "P 21 21 21")
    out = tmp_path / "cell.pdb"
    result = expand_to_unit_cell(asu, out)

    assert result.chain_ids == ["A", "B", "C", "D"]
    assert result.natoms == 12
    assert set(_fields(out, 21, 22)) == {"A", "B", "C", "D"}
    # atom order, residue names and serials run through the copies without gaps
    assert _fields(out, 17, 20) == ["ALA", "GLY", "SER"] * 4
    assert [int(s) for s in _fields(out, 6, 11)] == list(range(1, 13))


def test_expand_packs_every_copy_centroid_into_the_cell(tmp_path: Path):
    asu = _write_asu(tmp_path, "P 21 21 21")
    out = tmp_path / "cell.pdb"
    expand_to_unit_cell(asu, out)

    cell = gemmi.UnitCell(*CELL)
    copies = _coords(out)
    for start in range(0, len(copies), 3):
        fractional = [cell.fractionalize(gemmi.Position(*xyz)) for xyz in copies[start : start + 3]]
        for axis in ("x", "y", "z"):
            centroid = sum(getattr(f, axis) for f in fractional) / 3
            assert 0.0 <= centroid < 1.0


def test_expand_op_order_permutes_the_copies(tmp_path: Path):
    asu = _write_asu(tmp_path, "P 21 21 21")
    default = tmp_path / "default.pdb"
    swapped = tmp_path / "swapped.pdb"
    expand_to_unit_cell(asu, default)
    expand_to_unit_cell(asu, swapped, op_order=[1, 2, 4, 3])

    got, expected = _coords(swapped), _coords(default)
    assert got[:6] == expected[:6]
    assert got[6:9] == expected[9:12]  # chain C now holds the fourth operation
    assert got[9:12] == expected[6:9]


def test_expand_rejects_an_op_order_that_is_not_a_permutation(tmp_path: Path):
    asu = _write_asu(tmp_path, "P 21 21 21")
    with pytest.raises(CrystalSymmetryError, match="permutation"):
        expand_to_unit_cell(asu, tmp_path / "cell.pdb", op_order=[1, 2, 3])


def test_expand_honors_a_spacegroup_override(tmp_path: Path):
    asu = _write_asu(tmp_path, "P 1")
    result = expand_to_unit_cell(asu, tmp_path / "cell.pdb", spacegroup="P 2")
    assert result.spacegroup == "P 1 2 1"  # gemmi reports the full Hermann-Mauguin name
    assert result.natoms == 6


def test_expand_without_cryst1_raises(tmp_path: Path):
    asu = tmp_path / "nocell.pdb"
    asu.write_text(_atom(1, "N", "ALA", 1, (1.0, 2.0, 3.0)))
    with pytest.raises(CrystalSymmetryError, match="no CRYST1"):
        expand_to_unit_cell(asu, tmp_path / "cell.pdb")


def test_expand_with_an_unknown_spacegroup_raises(tmp_path: Path):
    asu = _write_asu(tmp_path, "P 1")
    with pytest.raises(CrystalSymmetryError, match="unknown space group"):
        expand_to_unit_cell(asu, tmp_path / "cell.pdb", spacegroup="Q 7")


def test_propagate_shifts_by_lattice_vectors_and_scales_cryst1(tmp_path: Path):
    cell = _write_asu(tmp_path, "P 1")
    out = tmp_path / "xtal.pdb"
    result = propagate_cell(cell, out, ix=2, iy=1, iz=1)

    assert (result.ncells, result.natoms, result.nresidues) == (2, 6, 6)
    assert result.numbering == "continuous"
    coords = _coords(out)
    assert coords[:3] == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    assert coords[3:] == [(x + CELL[0], y, z) for x, y, z in coords[:3]]
    cryst1 = out.read_text().splitlines()[0]
    assert float(cryst1[6:15]) == 2 * CELL[0]
    assert float(cryst1[15:24]) == CELL[1]
    assert [int(s) for s in _fields(out, 6, 11)] == list(range(1, 7))
    assert [int(s) for s in _fields(out, 22, 26)] == list(range(1, 7))


def test_propagate_replicates_with_z_fastest_like_proppdb(tmp_path: Path):
    cell = _write_asu(tmp_path, "P 1")
    out = tmp_path / "xtal.pdb"
    propagate_cell(cell, out, ix=2, iy=1, iz=2)

    first_atoms = _coords(out)[::3]  # one atom per replicated cell
    assert first_atoms == [
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0 + CELL[2]),
        (1.0 + CELL[0], 2.0, 3.0),
        (1.0 + CELL[0], 2.0, 3.0 + CELL[2]),
    ]


def test_propagate_per_cell_restarts_counters_and_tags_the_segid(tmp_path: Path):
    cell = _write_asu(tmp_path, "P 1")
    out = tmp_path / "xtal.pdb"
    result = propagate_cell(cell, out, ix=2, numbering="per-cell")

    assert result.numbering == "per-cell"
    assert [int(s) for s in _fields(out, 6, 11)] == [1, 2, 3, 1, 2, 3]
    assert [int(s) for s in _fields(out, 22, 26)] == [1, 2, 3, 1, 2, 3]
    assert _fields(out, 21, 22) == ["A"] * 3 + ["B"] * 3
    assert _fields(out, 72, 76) == ["0000"] * 3 + ["0001"] * 3


def test_propagate_auto_switches_to_per_cell_when_the_pdb_fields_overflow(tmp_path: Path):
    cell = _write_asu(tmp_path, "P 1")
    out = tmp_path / "xtal.pdb"
    result = propagate_cell(cell, out, ix=15, iy=15, iz=15)

    assert result.nresidues == 3 * 15**3 > 9999
    assert result.numbering == "per-cell"
    assert max(int(s) for s in _fields(out, 22, 26)) == 3


def test_propagate_rejects_counts_below_one(tmp_path: Path):
    cell = _write_asu(tmp_path, "P 1")
    with pytest.raises(CrystalSymmetryError, match=">= 1"):
        propagate_cell(cell, tmp_path / "xtal.pdb", ix=0)
