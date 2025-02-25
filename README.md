# ReproSchema to NDA Converter

Convert [NIMH-Minimal](https://www.repronim.org/nimh-minimal/#/) ReproSchema responses to [NDA (National Data Archive) Common Data Elements (CDE) format](https://nda.nih.gov/nda/nimh-common-data-elements). This tool helps standardize mental health assessment data collection by mapping ReproSchema responses to NDA's required formats.

## Features

- Converts ReproSchema response files to NDA CDE format
- Supports multiple mental health assessments (see [here](https://github.com/ReproNim/nimh-minimal)):
  - DSM-5 Cross-Cutting (Adult, Youth, Parent/Guardian)
  - GAD-7
  - PHQ-9
  - RCADS (Youth and Caregiver)
  - WHODAS
- Handles subject demographics data
- Maintains data consistency through template validation
- Combines multiple responses for the same assessment when needed
- Debug modes for troubleshooting

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/reproschema-mapping.git
cd reproschema-mapping

# Install dependencies
pip install -e .
```

## Usage

### Basic Usage

```bash
python scripts/process_mapping.py --response-dir /path/to/responses --cde-dir /path/to/cde --output-dir /path/to/output
```

### Debug Modes

For troubleshooting, use one of the debug modes:

```bash
# Process single file
python scripts/process_mapping.py --debug single

# Process all files with detailed logging
python scripts/process_mapping.py --debug full

# Process all files with summary statistics
python scripts/process_mapping.py --debug summary
```

## Directory Structure

```bash
.
├── data/
│   └── nda_cde/          # CDE definition and template files
│
├── scripts/
│   └── process_mapping.py # Main processing script
├── src/
│   └── rs2nda/          # Core conversion logic
└── tests/               # Test suite
```

## Running Tests

```bash
pytest tests/ -v --asyncio-mode=auto
```
