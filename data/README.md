# Data

The datasets are included in this repository as zip files. Extract them here before running any benchmarks:

```bash
cd data
unzip Flab-curate.zip
unzip poly.zip
```

## Datasets

- `Flab-curate/` — FLab curated antibody developability dataset (expression-based splits)
- `poly/` — VHH polyreactivity dataset

## Expected structure

```
data/
├── Flab-curate/
│   ├── Train_Random.csv
│   ├── Test_Random.csv
│   ├── Val_Random.csv
│   ├── Train_1H.csv
│   ├── Test_1H.csv
│   ├── Val_1H.csv
│   ├── Train_3H.csv
│   ├── Test_3H.csv
│   └── Val_3H.csv
└── poly/
    ├── vhh_splitA_novelty_train.csv
    ├── vhh_splitA_novelty_test.csv
    ├── vhh_splitA_novelty_val.csv
    ├── vhh_splitB_prevalence_train.csv
    ├── vhh_splitB_prevalence_test.csv
    └── vhh_splitB_prevalence_val.csv
```
