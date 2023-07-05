




def load_mol_input(dir: str, canonical_SMILES: bool) -> list[int]:
    """
    loads the dataset based on either SDF or SMILES format.
    Args:
        dir (str): dir to file (incl filename)
        canonical_SMILES (boolean): Generate canonical SMILES - default True
    Returns:
        list (CDPKit Mol)
        list (SMILES)
    """

    print(dir)
    print(canonical_SMILES)

    return list(1,2,3)


load_mol_input("aha",True)