import os
from pathlib import Path
from typing import Any, Iterable

from nerdd_module import Model, Mol
from nerdd_module.preprocessing import Sanitize
from torch_geometric.loader import DataLoader

from awesom.dataset import SOMDataset
from awesom.model import SOMPredictor, predict_ensemble
from awesom.metrics import THRESHOLD

MODEL_DIRECTORY = Path(os.environ["AWESOM_MODEL_DIRECTORY"])


def load_models() -> list[SOMPredictor]:
    models = []
    for model_path in sorted(MODEL_DIRECTORY.glob("model_*")):
        checkpoint_path = model_path / "checkpoints" / "best_model.ckpt"
        models.append(SOMPredictor.load(str(checkpoint_path)))
    return models


class AweSOMModel(Model):
    def __init__(self, preprocessing_steps=[Sanitize()]):
        self.model_ensemble = load_models()

    def _predict_mols(self, mols: list[Mol], **kwargs: Any) -> Iterable[dict]:
        data = [
            SOMDataset.mol_to_data(mol, soms=[], mol_id=mol_id, description="")
            for mol_id, mol in enumerate(mols)
        ]

        predictions = predict_ensemble(DataLoader(data), self.model_ensemble)

        for mol_id, atom_id, probability, u_ale, u_epi, u_tot in zip(
            predictions.mol_ids,
            predictions.atom_ids,
            predictions.get_probabilities().mean(dim=0),
            *predictions.get_uncertainties(),
        ):
            yield {
                "mol_id": mol_id,
                "atom_id": atom_id,
                "prediction": probability,
                "prediction_binary": probability > THRESHOLD,
                # "ranking": 0,
                "aleatoric_uncertainty": u_ale,
                "epistemic_uncertainty": u_epi,
                "total_uncertainty": u_tot,
            }
