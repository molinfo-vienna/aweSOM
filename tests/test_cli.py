from pathlib import Path
from tempfile import TemporaryDirectory

from awesom._cli import hyperparameters, predict, train

data = Path(__file__).parent / "test_data"


def test_train_then_predict():
    with TemporaryDirectory() as tmpdir:
        train(
            input_path=data / "train_data_small.sdf",
            output_path=Path(tmpdir) / "models",
            config_path=data / "example_hparams.yaml",
        )
        predict(
            input_path=data / "test_data_small.sdf",
            models_path=Path(tmpdir) / "models",
            output_path=Path(tmpdir) / "predictions",
        )


def test_hyperparameter_search():
    with TemporaryDirectory() as tmpdir:
        hyperparameters(
            input_path=data / "train_data_small.sdf",
            output_path=Path(tmpdir) / "hp_search",
            num_folds=2,
        )
