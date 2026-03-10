#!/usr/bin/env python3
"""
Kremer-Grest Polymer Model Analysis Script

Calculates entropy-related quantities from LAMMPS snapshot files using:
- Coordinate fluctuations (QQ)
- Momentum fluctuations (QP)
- PCA-based correlation analysis (QC)
- Energy exchange/thermal properties (QT)

Author: [Your Name]
License: MIT

Usage:
    python analyze_kg.py --snap-dir T1.0/2_restart --output-dir results
    python analyze_kg.py --snap-dir T1.0/2_restart --start 1000 --nu 10,20,40
    python analyze_kg.py --help

Input:
    LAMMPS snapshot files (lammps_snap.*) with:
    - Scaled coordinates: xs, ys, zs
    - Image flags: ix, iy, iz
    - Velocities: vx, vy, vz (optional)
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

__version__ = "1.0.0"

# Constants
DEFAULT_START_FRAME = 1000
DEFAULT_ATOMS_PER_CHAIN = 200
DEFAULT_NU_VALUES = [40, 20, 10]
ZERO_THRESHOLD = 1e-10

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class SnapshotData:
    """Container for LAMMPS snapshot data."""
    timestep: int
    box_lengths: NDArray[np.float64]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]


@dataclass
class FluctuationResults:
    """Container for fluctuation analysis results."""
    qq: NDArray[np.float64]  # Coordinate fluctuations
    qp: NDArray[np.float64]  # Momentum fluctuations
    qc: NDArray[np.float64]  # Correlation (PCA)
    qt: NDArray[np.float64]  # Energy exchange
    box_lengths: NDArray[np.float64]
    nu: int
    n_groups: int


def read_lammps_snapshot(filepath: str) -> SnapshotData:
    """
    Read a single LAMMPS snapshot file.

    Args:
        filepath: Path to the LAMMPS snapshot file

    Returns:
        SnapshotData containing timestep, box dimensions, positions, and velocities

    Raises:
        ValueError: If file format is invalid
        FileNotFoundError: If file does not exist
    """
    with open(filepath, "r") as f:
        # TIMESTEP
        line = f.readline()
        if not line.startswith("ITEM: TIMESTEP"):
            raise ValueError(f"Expected ITEM: TIMESTEP, got: {line.strip()}")
        timestep = int(f.readline().strip())

        # NUMBER OF ATOMS
        f.readline()
        n_atoms = int(f.readline().strip())

        # BOX BOUNDS
        f.readline()
        bounds = []
        for _ in range(3):
            parts = f.readline().split()
            bounds.append([float(parts[0]), float(parts[1])])
        bounds = np.array(bounds)
        box_lengths = bounds[:, 1] - bounds[:, 0]

        # ATOMS header
        header_line = f.readline()
        columns = header_line.split()[2:]  # Skip "ITEM:" and "ATOMS"

        # Find column indices
        id_idx = columns.index("id")
        xs_idx = columns.index("xs")
        ys_idx = columns.index("ys")
        zs_idx = columns.index("zs")

        # Find image flag indices (ix, iy, iz)
        has_image = "ix" in columns
        if has_image:
            ix_idx = columns.index("ix")
            iy_idx = columns.index("iy")
            iz_idx = columns.index("iz")

        # Check if velocity columns exist
        has_velocity = "vx" in columns
        if has_velocity:
            vx_idx = columns.index("vx")
            vy_idx = columns.index("vy")
            vz_idx = columns.index("vz")

        # Read atom data
        atoms = []
        for _ in range(n_atoms):
            parts = f.readline().split()
            atom_id = int(parts[id_idx])
            xs = float(parts[xs_idx])
            ys = float(parts[ys_idx])
            zs = float(parts[zs_idx])

            # Get image flags (default 0 if not present)
            if has_image:
                ix = int(parts[ix_idx])
                iy = int(parts[iy_idx])
                iz = int(parts[iz_idx])
            else:
                ix, iy, iz = 0, 0, 0

            # Calculate real coordinates: (scaled + image) * box_length
            x_real = (xs + ix) * box_lengths[0]
            y_real = (ys + iy) * box_lengths[1]
            z_real = (zs + iz) * box_lengths[2]

            if has_velocity:
                vx = float(parts[vx_idx])
                vy = float(parts[vy_idx])
                vz = float(parts[vz_idx])
                atoms.append((atom_id, x_real, y_real, z_real, vx, vy, vz))
            else:
                atoms.append((atom_id, x_real, y_real, z_real, 0.0, 0.0, 0.0))

        # Sort by atom ID
        atoms.sort(key=lambda x: x[0])

        positions = np.array([[a[1], a[2], a[3]] for a in atoms])
        velocities = np.array([[a[4], a[5], a[6]] for a in atoms])

    return SnapshotData(
        timestep=timestep,
        box_lengths=box_lengths,
        positions=positions,
        velocities=velocities,
    )


def analyze_fluctuations(
    snap_dir: str,
    start: int,
    stop: Optional[int],
    nu: int,
    max_atoms: Optional[int] = None,
    atoms_per_chain: int = DEFAULT_ATOMS_PER_CHAIN,
) -> FluctuationResults:
    """
    Read LAMMPS snapshot files and calculate fluctuation quantities.

    Args:
        snap_dir: Directory containing lammps_snap.* files
        start: Start frame index
        stop: End frame index (None for all)
        nu: Number of particles to group together
        max_atoms: Maximum number of atoms to use (None for all)
        atoms_per_chain: Number of atoms per polymer chain

    Returns:
        FluctuationResults containing QQ, QP, QC, QT values

    Raises:
        ValueError: If no snapshot files found
    """
    # Find all snapshot files
    pattern = os.path.join(snap_dir, "lammps_snap.*")
    files = glob.glob(pattern)

    # Extract timesteps and sort
    file_timesteps = []
    for f in files:
        match = re.search(r"lammps_snap\.(\d+)$", f)
        if match:
            file_timesteps.append((int(match.group(1)), f))
    file_timesteps.sort(key=lambda x: x[0])

    # Select frames
    file_timesteps = file_timesteps[start:stop]

    if len(file_timesteps) == 0:
        raise ValueError(f"No snapshot files found in {snap_dir}")

    logger.info("Reading %d frames from %s", len(file_timesteps), snap_dir)

    # Read first file to get dimensions
    first_snap = read_lammps_snapshot(file_timesteps[0][1])
    n_atoms_total = first_snap.positions.shape[0]
    box_lengths = first_snap.box_lengths

    # Limit atoms if specified
    if max_atoms is not None and n_atoms_total > max_atoms:
        n_atoms = max_atoms
        logger.info("Using %d atoms (from %d total)", max_atoms, n_atoms_total)
    else:
        n_atoms = n_atoms_total

    # Ensure n_atoms is divisible by NU
    n_atoms = (n_atoms // nu) * nu
    n_groups = n_atoms // nu

    logger.info("Grouping %d atoms into %d groups of %d", n_atoms, n_groups, nu)

    # Calculate number of chains
    n_chains = n_atoms_total // atoms_per_chain
    logger.info("Detected %d chains with %d atoms each", n_chains, atoms_per_chain)
    logger.info("Removing chain center-of-mass fluctuations")

    # Read all frames
    X = []  # [n_frames, n_groups, 6*NU]

    for i, (ts, filepath) in enumerate(file_timesteps):
        if (i + 1) % 100 == 0 or i == 0:
            logger.info("Reading frame %d/%d...", i + 1, len(file_timesteps))

        snap = read_lammps_snapshot(filepath)
        positions = snap.positions.copy()
        velocities = snap.velocities.copy()

        # Remove chain center-of-mass motion
        for ic in range(n_chains):
            start_idx = ic * atoms_per_chain
            end_idx = (ic + 1) * atoms_per_chain
            chain_com_pos = np.mean(positions[start_idx:end_idx], axis=0)
            positions[start_idx:end_idx] -= chain_com_pos
            chain_com_vel = np.mean(velocities[start_idx:end_idx], axis=0)
            velocities[start_idx:end_idx] -= chain_com_vel

        # Select atoms and group
        y = np.zeros((n_groups, 6 * nu))
        for ia in range(n_groups):
            for ip in range(nu):
                atom_idx = ia * nu + ip
                if atom_idx < n_atoms_total:
                    y[ia, ip * 3 : ip * 3 + 3] = positions[atom_idx]
                    y[ia, ip * 3 + 3 * nu : ip * 3 + 3 * nu + 3] = velocities[atom_idx]
        X.append(y)

    X = np.array(X)
    nshot = X.shape[0]
    logger.info("Data shape: %s", X.shape)
    logger.info("Coordinates unwrapped via image flags, chain COM removed")

    # Centering
    Xave = np.average(X, axis=0)
    X = X - Xave
    logger.info("Data normalized")

    # Calculate fluctuation quantities
    logger.info("Calculating fluctuations with PCA...")
    pca = PCA(n_components=min(nu * 6, nshot - 1))

    qq = np.zeros(n_groups)
    qp = np.zeros(n_groups)
    qc = np.zeros(n_groups)
    qt = np.zeros(n_groups)

    for ia in range(n_groups):
        if (ia + 1) % 500 == 0:
            logger.info("Processing group %d/%d...", ia + 1, n_groups)

        x = X[:, ia, :]
        xdev = np.std(x, axis=0)
        xdev[xdev == 0] = ZERO_THRESHOLD

        # Sum log of standard deviations
        for ip in range(nu * 3):
            if xdev[ip] > 0:
                qq[ia] += np.log(xdev[ip])
            if xdev[ip + nu * 3] > 0:
                qp[ia] += np.log(xdev[ip + nu * 3])

        # PCA analysis
        x_norm = x / xdev
        try:
            xpca = pca.fit_transform(x_norm)
            xpcadev = np.std(xpca, axis=0)
            xpcadev[xpcadev == 0] = ZERO_THRESHOLD

            for ip in range(len(xpcadev)):
                if xpcadev[ip] > 0:
                    qc[ia] += np.log(xpcadev[ip])

            xpca_scaled = xpca / xpcadev
            xpca_scaled2 = np.sum(xpca_scaled * xpca_scaled, axis=1)
            std_val = np.std(xpca_scaled2)
            if std_val > 0:
                qt[ia] = np.log(std_val)
        except Exception as e:
            logger.debug("PCA failed for group %d: %s", ia, e)

    return FluctuationResults(
        qq=qq,
        qp=qp,
        qc=qc,
        qt=qt,
        box_lengths=box_lengths,
        nu=nu,
        n_groups=n_groups,
    )


def save_results(
    results: FluctuationResults,
    output_path: str,
    snap_dir: str,
    start: int,
    stop: Optional[int],
    max_atoms: Optional[int],
    atoms_per_chain: int,
) -> None:
    """Save fluctuation analysis results to CSV file."""
    with open(output_path, "w") as f:
        f.write("# Fluctuation Analysis Results\n")
        f.write(f"# Directory: {snap_dir}\n")
        f.write(f"# Frames: {start} to {stop}\n")
        f.write(f"# Max atoms: {max_atoms}\n")
        f.write(f"# Atoms per chain: {atoms_per_chain}\n")
        f.write("# Chain center-of-mass fluctuations removed\n")
        f.write(f"# NU: {results.nu}\n")
        f.write(f"# Number of groups: {results.n_groups}\n")
        f.write(f"# QQ mean: {np.mean(results.qq):.6f}, std: {np.std(results.qq):.6f}\n")
        f.write(f"# QP mean: {np.mean(results.qp):.6f}, std: {np.std(results.qp):.6f}\n")
        f.write(f"# QC mean: {np.mean(results.qc):.6f}, std: {np.std(results.qc):.6f}\n")
        f.write(f"# QT mean: {np.mean(results.qt):.6f}, std: {np.std(results.qt):.6f}\n\n")
        f.write("GroupID,QQ,QP,QC,QT\n")
        for i in range(results.n_groups):
            f.write(f"{i},{results.qq[i]:.6f},{results.qp[i]:.6f},{results.qc[i]:.6f},{results.qt[i]:.6f}\n")


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Kremer-Grest polymer model from LAMMPS snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --snap-dir T1.0/2_restart
  %(prog)s --snap-dir T1.0/2_restart --start 1000 --nu 10,20,40
  %(prog)s --snap-dir T1.0/2_restart --output-dir results --atoms-per-chain 200
        """,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--snap-dir",
        type=str,
        default="T1.0/2_restart",
        help="Directory containing lammps_snap.* files (default: T1.0/2_restart)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for CSV files (default: current directory)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=DEFAULT_START_FRAME,
        help=f"Start frame index (default: {DEFAULT_START_FRAME})",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="End frame index (default: None, use all frames)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Maximum number of atoms to use (default: None, use all)",
    )
    parser.add_argument(
        "--atoms-per-chain",
        type=int,
        default=DEFAULT_ATOMS_PER_CHAIN,
        help=f"Number of atoms per polymer chain (default: {DEFAULT_ATOMS_PER_CHAIN})",
    )
    parser.add_argument(
        "--nu",
        type=str,
        default=",".join(map(str, DEFAULT_NU_VALUES)),
        help=f"Comma-separated list of NU values (default: {','.join(map(str, DEFAULT_NU_VALUES))})",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="T1.0",
        help="Label for output files (default: T1.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args(args)


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for the analysis script.

    Args:
        args: Command line arguments (None to use sys.argv)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parsed_args = parse_args(args)

    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse NU values
    nu_values = [int(x.strip()) for x in parsed_args.nu.split(",")]

    logger.info("=" * 70)
    logger.info("Kremer-Grest Polymer Analysis v%s", __version__)
    logger.info("=" * 70)
    logger.info("Data directory: %s", parsed_args.snap_dir)
    logger.info("Output directory: %s", parsed_args.output_dir)
    logger.info("Frames: %s to %s", parsed_args.start, parsed_args.stop)
    logger.info("Max atoms: %s", parsed_args.max_atoms)
    logger.info("Atoms per chain: %s", parsed_args.atoms_per_chain)
    logger.info("NU values: %s", nu_values)

    # Validate input directory
    if not os.path.isdir(parsed_args.snap_dir):
        logger.error("Snapshot directory not found: %s", parsed_args.snap_dir)
        return 1

    # Create output directory if needed
    os.makedirs(parsed_args.output_dir, exist_ok=True)

    # Analyze with different group sizes
    for nu in nu_values:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Analysis with NU = %d (grouping %d atoms)", nu, nu)
        logger.info("=" * 70)

        try:
            results = analyze_fluctuations(
                parsed_args.snap_dir,
                parsed_args.start,
                parsed_args.stop,
                nu,
                max_atoms=parsed_args.max_atoms,
                atoms_per_chain=parsed_args.atoms_per_chain,
            )
        except ValueError as e:
            logger.error("Analysis failed: %s", e)
            return 1

        # Print summary statistics
        logger.info("Results for NU=%d:", nu)
        logger.info("  QQ (coordinates): mean=%.4f, std=%.4f", np.mean(results.qq), np.std(results.qq))
        logger.info("  QP (momentum):    mean=%.4f, std=%.4f", np.mean(results.qp), np.std(results.qp))
        logger.info("  QC (correlation): mean=%.4f, std=%.4f", np.mean(results.qc), np.std(results.qc))
        logger.info("  QT (energy):      mean=%.4f, std=%.4f", np.mean(results.qt), np.std(results.qt))

        # Save results
        output_path = os.path.join(
            parsed_args.output_dir, f"fluctuation_analysis_{parsed_args.label}_NU{nu}.csv"
        )
        save_results(
            results,
            output_path,
            parsed_args.snap_dir,
            parsed_args.start,
            parsed_args.stop,
            parsed_args.max_atoms,
            parsed_args.atoms_per_chain,
        )
        logger.info("Results saved to: %s", output_path)

    logger.info("")
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
