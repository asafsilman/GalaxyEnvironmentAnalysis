# Galaxy Environment Analysis (GEA)

This repository contains the codebase for analysing the surrounding environment of galaxies using tensorflow

## Project Structure

The structure of the project is based on [This Project](https://drivendata.github.io/cookiecutter-data-science/#data-is-immutable)

```
├── data
│   ├── raw
│   ├── interim 
│   └── processed
├── logs
├── models
├── notebooks
├── reports
├── src
├── gea.py
└── README.md
```

Data for this project is treated as a Directed Acyclic Graph (DAG), it is processed from the the raw directory to the interim directory to a format which is more compatiable with tensorflow, the data is then split into these directories `processed/<model>/train|test|validation/`.

The config for this project specifies which raw files are used for each model, and how they are processed.

Once the data is processed and training has begun, the model file and weights are stored under `models/`, and the tensorboard logs are stored in `logs/`.

Any jupyter notebooks which are mainly used for experimentation and generating visualisations are stored under `notebooks/`

Reports such as the research proposal and thesis paper are stored under `reports/`.


The source code is stored under `src/`

## CLI

The cli tool can be helpful in automating most of the project tasks, such as data preperation, data splitting, training, and cleaning directory.

**Cleanup project directories:** `python gea.py clean`

**Prepare data from config:** `python gea.py prepare_data <modelname>`

**Prepare and split data from config** `python gea.py run-experiment <modelname>`

