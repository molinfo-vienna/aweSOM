
import CDPL.Chem as Chem
import CDPL.MolProp as MolProp # TS: added
import os

ELEM_LIST =[5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53,"OTHER"]
HYBRIDIZATION_TYPE = ["SP", 
                      "SP2", 
                      "SP3", 
                      "OTHER"]
FORMAL_CHARGE = [-1,0,1,"OTHER"]
RING_SIZE = [0,3,4,5,6,7,8,"OTHER"]
TOTAL_DEGREE = [1,2,3,4,"OTHER"]
TOTAL_VALENCE = [1,2,3,4,5,6,"OTHER"]
RING_COUNT = [0,1,2,3,4,5,"OTHER"]

# TS: note that STEREOZ is equivalent to STEREOCIS and STEREOTRANS equivalent to STEREOE!!
BOND_STEREO = ["STEREONONE",
               "STEREOANY", 
               "STEREOZ", 
               "STEREOE", 
               "STEREOCIS", 
               "STEREOTRANS",
               "OTHER",
               ]
BOND_TYPE = ["UNSPECIFIED", 
             "SINGLE", 
             "DOUBLE", 
             "TRIPLE",
             "OTHER",
             ]

### Ich lade ein SDF oder SMILES mit mehreren Entries und möchte die Hydrogens entfernen und
### sollten nicht schon SMILES geladen werden, möchte ich aus den Chem.BasicMolecule das entsprechende
### SMILES generieren und ein paar andere properties aus mols/bonds um sie dann in eine andere liste zu geben

# 1 ------------>

#extention = ".sdf" # or .smiles
#dir = "/path/to/filename"
#filename = "file"

#ipt_handler = Chem.MoleculeIOManager.getInputHandlerByFileExtension(extention.lower()) # .sdf
#reader = ipt_handler.createReader(os.path.join(dir, filename+extention))

# ----------- TS:
filename = '/path/to/filename/file.sdf'
name_and_ext = os.path.splitext(filename)

ipt_handler = Chem.MoleculeIOManager.getInputHandlerByFileExtension(name_and_ext[1][1:]) # extension immer ohne leading '.', case egal!
reader = ipt_handler.createReader(filename)

# 1 <--------------

# 2 -------------->

#mol = Chem.BasicMolecule()
#mols = list()
#smiles = list()

#while reader.read(mol):    
    # remove Hs here
    # Chem.removeH(mol)
    # generate SMILES
    # smile = Chem.getSmiles(mol)
    # smiles.append(smile)
    # calculate fraction of rotatable Bonds
    # num_rotatable_bonds = Chem.CalcNumRotatableBonds(mol)
    # num_bonds = mol.GetNumBonds()
    # mols.append(mol)

# ----------- TS:
mols = list()
smiles = list()

while True:
    mol = Chem.BasicMolecule()

    if not reader.read(mol):
        break
    
    Chem.calcImplicitHydrogenCounts(mol, False)  # calculate implicit hydrogen counts and set corresponding property for all atoms
    Chem.perceiveHybridizationStates(mol, False) # perceive atom hybridization states and set corresponding property for all atoms
    Chem.perceiveSSSR(mol, False)                # perceive smallest set of smallest rings and store as Chem.MolecularGraph property
    Chem.setRingFlags(mol, False)                # perceive cycles and set corresponding atom and bond properties
    Chem.setAromaticityFlags(mol, False)         # perceive aromaticity and set corresponding atom and bond properties
    Chem.makeHydrogenDeplete(mol)                # remove explicit hydrogens
    
    smile = Chem.generateSMILES(mol, True)       # True -> canonical SMILES
    smiles.append(smile)
    
    num_rotatable_bonds = MolProp.getRotatableBondCount(mol)
    num_bonds = mol.numBonds

    mols.append(mol)
    
# 2 <---------------------
    
    # calculate mol features
    #mol.GetNumAtoms()        # TS: which count does RDKit report here?
    mol.numAtoms              # TS alt. 1: heavy atom count only (now after removal of Hs)!
    MolProp.getAtomCount(mol) # TS alt. 2: total atom count = heavy atoms + hydrogens (impl. and/or expl.)
    
    #mol.CalcNumHBA()
    MolProp.getHBondAcceptorAtomCount(mol) # TS
    #mol.CalcNumHBD()
    MolProp.getHBondDonorAtomCount(mol) # TS
    
    #len(mol.GetAromaticAtoms())
    MolProp.getAromaticAtomCount(mol) # TS
    
    #mol.CalcNumRings() # as in RING_COUNT TS: I assume the number of rings in the SSSR is meant here?

    num_rings = MolProp.calcCyclomaticNumber(mol) # TS

    if num_rings not in RING_COUNT: # TS
        num_rings = 'OTHER'         # TS
    
    # calculate atom features
    for atom in mol.atoms:
        #atom.GetAtomicNum() # as in ELEM_LIST

        atomic_no = Chem.getType(atom) # TS

        if atomic_no not in ELEM_LIST: # TS
            atomic_no = 'OTHER'        # TS
        
        #atom.GetFormalCharge() # as in FORMAL_CHARGE

        form_chg = Chem.getFormalCharge(atom) # TS

        if form_chg not in FORMAL_CHARGE:     # TS
            form_chg = 'OTHER'                # TS
       
        #atom.GetHybridization() # as in HYBRIDIZATION_TYPE

        hyb_state = Chem.getHybridizationState(atom)   # TS

        if hyb_state == Chem.HybridizationState.SP1:   # TS
            hyb_state = 'SP'                           # TS
        elif hyb_state == Chem.HybridizationState.SP2: # TS
            hyb_state = 'SP2'                          # TS
        elif hyb_state == Chem.HybridizationState.SP3: # TS
            hyb_state = 'SP3'                          # TS
        else:                                          # TS
            hyb_state = 'OTHER'                        # TS
        
        #atom.atm_ring_length() # as in RING_SIZE TS: this is not a unique property, an atom can be in multiple rings with different sizes! Which size to report then?

        rsize = Chem.getSizeOfSmallestContainingFragment(atom, Chem.getSSSR(mol)) # TS: the size of the smallest containing ring is reported here

        if rsize not in RING_SIZE: # TS
            rsize = 'OTHER'        # TS
        
        #atom.GetIsAromatic()
        Chem.getAromaticityFlag(atom) # TS
        
        #atom.GetTotalDegree() # as in TOTAL_DEGREE

        degree = MolProp.getBondCount(atom) # TS: degree including Hs! MolProp.getHeavyBondCount(atom) reports degree without Hs

        if degree not in TOTAL_DEGREE:      # TS
            degree = 'OTHER'                # TS
                                      
        #atom.GetTotalValence() # as in TOTAL_VALENCE

        valence = MolProp.calcValence(atom, mol) # TS

        if valence not in TOTAL_VALENCE:         # TS
            valence = 'OTHER'                    # TS
        
        #atom.GetSymbol() # for ["F", "Cl", "Br", "I", "N", "O", "S"]
        Chem.getSymbol(atom)        # TS
        
        #for atom_idx in range(mol.GetNumAtoms()):
        #    atom = mol.GetAtomWithIdx(atom_idx)

        for atom_idx in range(mol.numAtoms): # TS
            atom = mol.getAtom(atom_idx)     # TS

    elec_sys_list = perceivePiElectronSystems(mol) # TS: for bond.GetIsConjugated() replacement
    Chem.calcBondStereoDescriptors(mol, False)     # TS: for bond.GetStereo() replacement
    
    # calculate bond features
    for bond in mol.bonds:
        #bond.IsInRing()
        Chem.getRingFlag(bond) # TS
        
        #bond.GetIsConjugated() # TS: The RDKit docu does not tell what 'conjugated' exactly means here, I assume part of a pi electron system...

        is_conj = False                  # TS
        
        for elec_sys in elec_sys_list:   # TS
            if elec_sys.numAtoms < 3:    # TS
                continue                 # TS 

            if elec_sys.containsAtom(bond.getBegin()) and elec_sys.containsAtom(bond.getEnd()): # TS
                is_conj = True           # TS
                break                    # TS
        
        #bond.GetIsAromatic()
        Chem.getAromaticityFlag(bond) # TS
        
        #bond.GetStereo() # as in BOND_STEREO

        stereo = Chem.getStereoDescriptor(bond).getConfiguration() # TS

        if stereo == Chem.BondConfiguration.NONE:     # TS
            stereo = "STEREONONE"                     # TS
        elif stereo == Chem.BondConfiguration.E:      # TS
            stereo = "STEREOE"                        # TS
        elif stereo == Chem.BondConfiguration.Z:      # TS
            stereo = "STEREOZ"                        # TS
        elif stereo == Chem.BondConfiguration.EITHER: # TS
            stereo = "STEREOANY"                      # TS
        else:                                         # TS
            stereo = "OTHER"                          # TS
            
        #bond.GetBondTypeAsDouble() # as in BOND_TYPE <- TS: this not what GetBondTypeAsDouble() returns!!!

        dbl_type = float(Chem.getOrder(bond)) # TS

        if Chem.getAromaticityFlag(bond):     # TS
            dbl_type = 1.5                    # TS
        
        #bond.GetBeginAtomIdx()
        mol.getAtomIndex(bond.getBegin()) # TS
        
        #bond.GetEndAtomIdx()
        mol.getAtomIndex(bond.getEnd())  # TS
        
    # calculate ring stuff
    #rings = mol.GetRingInfo().AtomRings()
    #for ring in rings:
    #    if atom_idx in ring:
    #        atm_ring_length = len(ring)

    rings = Chem.getSSSR(mol)    # TS

    for ring in rings:           # TS
        rlen = ring.numAtoms     # TS: length is not the most appropriate word for the size of a rings ;-)
        for atom in ring.atoms:  # TS
            pass                 # TS: do something here
