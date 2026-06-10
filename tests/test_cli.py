from pathlib import Path
from tempfile import TemporaryDirectory

from awesom._cli import predict, train

root = Path(__file__).parent


def test_train():
    with TemporaryDirectory() as tmpdir:
        train(
            input_path=root / "test_data/train",
            output_path=Path(tmpdir),
            config_path=root / "test_output/cv_hp_search/best_hparams.yaml",
        )


def test_predict():
    with TemporaryDirectory() as tmpdir:
        predict(
            input_path=root / "test_data/test",
            models_path=root / "test_output/model",
            output_path=Path(tmpdir),
        )
