import os
from pathlib import Path
from tempfile import TemporaryDirectory

from rdkit import Chem

from awesom._cli import train

data = Path(__file__).parent / "test_data"

IMATINIB = Chem.MolFromSmiles(
    "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
)


def test_train_then_nerdd():
    with TemporaryDirectory() as tmpdir:
        train(
            input_path=data / "train_data_small.sdf",
            output_path=Path(tmpdir) / "models",
            config_path=data / "example_hparams.yaml",
        )

        # Hack to test AweSOMModel without modifications
        os.environ["AWESOM_MODEL_DIRECTORY"] = str(Path(tmpdir) / "models")
        from awesom._nerdd import AweSOMModel

        model = AweSOMModel()
        predictions = model._predict_mols([IMATINIB])

        for prediction in predictions:
            assert isinstance(prediction, dict)
