import torch

def compute_ranking(y_hat, mol_id):
    ranking = torch.cat(
        [
            torch.argsort(
                torch.argsort(
                    torch.index_select(
                        y_hat[:, 1], 0, torch.where(mol_id == mid)[0]
                    ),
                    dim=0,
                    descending=True,
                ),
                dim=0,
                descending=False,
            )
            for mid in list(dict.fromkeys(mol_id.tolist())) # This is a somewhat complicated way to get an ordered set, but it works...
        ]
    )
    return ranking