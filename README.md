# aweSOM

[![Code quality checks status](https://img.shields.io/github/actions/workflow/status/molinfo-vienna/aweSOM/check.yml?label=check)](https://github.com/molinfo-vienna/aweSOM/actions/workflows/check.yml)
[![Tests status](https://img.shields.io/github/actions/workflow/status/molinfo-vienna/aweSOM/test.yml?label=test)](https://github.com/molinfo-vienna/aweSOM/actions/workflows/test.yml)

A model predicting sites-of-metabolism (SOMs) in xenobiotics with aleatoric and epistemic uncertainty estimation.

Paper: https://pubs.acs.org/doi/10.1021/acs.jcim.5c00762

## Online prediction

We provide AweSOM models deployed on the [NERDD](https://nerdd.univie.ac.at/awesom) webserver, freely available for non-commercial research.
If you need to predict SOMs for a very large number of molecules, we instead recommend you install and run our software locally.

## Installation

We recommend using [`uv`](https://docs.astral.sh/uv/) for running this project. After having installed `uv` you can simply clone this repository and use it to run the `awesom` command.

```sh
git clone https://github.com/molinfo-vienna/aweSOM.git
cd aweSOM
uv run awesom --help
```

The `awesom` command provides self-documenting subcommands for training aweSOM model ensembles, obtaining predictions, calculating prediction metrics and performing hyperparameter searches.

For example to predict SOMs for a set of molecule stored in a SD file:

```sh
uv run awesom predict -i INPUT.sdf -m ./models -o OUTPUT.csv
```

## Models

Models are available...

## License

This project is licensed under the MIT license.
