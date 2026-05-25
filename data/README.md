# Dataset Access

## MIRACL (Training)

```bash
# Download from HuggingFace
pip install datasets
python -c "from datasets import load_dataset; load_dataset('miracl/miracl', 'en')"
```

## Cross-Distribution Evaluation

### FEVER
```bash
python -c "from datasets import load_dataset; load_dataset('fever', 'v1.0')"
```

### HotpotQA
```bash
python -c "from datasets import load_dataset; load_dataset('hotpot_qa', 'fullwiki')"
```

### SciFact
```bash
python -c "from datasets import load_dataset; load_dataset('allenai/scifact')"
```

## PDFM CONUS-27 (Geospatial)

Ground truth variables for 35,000 US zip codes from Population Dynamics Foundation Model.

Source: https://github.com/google-research/population-dynamics

The benchmark CSV includes:
- Zip code identifiers
- 27 demographic/economic variables
- Ground truth labels for evaluation

Contact authors for PDFM embedding access (requires approval).
