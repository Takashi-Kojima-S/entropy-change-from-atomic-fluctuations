# KG Polymer Analysis

Kremer-Grest polymer model analysis tool for LAMMPS simulation data.

## Overview

This tool calculates entropy-related quantities from LAMMPS snapshot files:

- **QQ**: Coordinate fluctuations (log of position standard deviations)
- **QP**: Momentum fluctuations (log of velocity standard deviations)
- **QC**: PCA-based correlation analysis
- **QT**: Energy exchange / thermal properties

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run with sample data (included in repository)
python analyze_kg.py --snap-dir sample_data/T1.0 --start 0 --output-dir results

# Basic usage
python analyze_kg.py --snap-dir T1.0

# Specify output directory and parameters
python analyze_kg.py --snap-dir T1.0 --output-dir results --start 1000

# Multiple NU values
python analyze_kg.py --snap-dir T1.0 --nu 10,20,40

# Show help
python analyze_kg.py --help
```

## Sample Data

The `sample_data/T1.0/` directory contains sample LAMMPS simulation data (T=1.0) for testing:

- **Snapshot files** (last 5 timesteps of equilibrated MD trajectory):
  - `lammps_snap.199920000` ~ `lammps_snap.200000000`
- **LAMMPS input files**:
  - `in.data`: Initial atom configuration
  - `in.in`: LAMMPS input script
  - `in.restart`: Restart file

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--snap-dir` | `T1.0/2_restart` | Directory containing `lammps_snap.*` files |
| `--output-dir` | `.` | Output directory for CSV files |
| `--start` | `1000` | Start frame index |
| `--stop` | `None` | End frame index (None = use all) |
| `--max-atoms` | `None` | Maximum atoms to use (None = use all) |
| `--atoms-per-chain` | `200` | Number of atoms per polymer chain |
| `--nu` | `40,20,10` | Comma-separated list of NU values |
| `--label` | `T1.0` | Label for output files |
| `-v, --verbose` | `False` | Enable verbose output |

## Input File Format

LAMMPS snapshot files (`lammps_snap.*`) with the following columns:

- `id`: Atom ID
- `xs`, `ys`, `zs`: Scaled coordinates
- `ix`, `iy`, `iz`: Image flags (optional)
- `vx`, `vy`, `vz`: Velocities (optional)

## Output

CSV files with the following format:
- `fluctuation_analysis_{label}_NU{nu}.csv`

Each file contains:
- Header with analysis parameters
- Columns: `GroupID`, `QQ`, `QP`, `QC`, `QT`

## License

MIT License
