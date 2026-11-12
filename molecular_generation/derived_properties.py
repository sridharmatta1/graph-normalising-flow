"""Per-molecule properties that RDKit computes exactly from the molecular
graph, for conditioning and for checking generated molecules.

QM9's stored properties (mu, homo, lumo, gap, ...) are DFT results on a 3D
geometry and cannot be recomputed from a generated graph, so conditioning on
one of them can only be checked through a separate predictor. The properties
here can: the same function gives the training label of a QM9 molecule and
the achieved value of a generated one.

Both go through qm9_chem.graph_to_mol, so a training label and a generated
molecule's value are computed from an identical molecule representation.
Free of TensorFlow and absl so it can be tested without the cluster
environment.
"""

from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors

from qm9_chem import graph_to_mol

DERIVED_PROPERTIES = {
    'logp': Crippen.MolLogP,
    'tpsa': rdMolDescriptors.CalcTPSA,
}


def mol_property(mol, name):
    """Value of a derived property for an RDKit molecule."""
    if name not in DERIVED_PROPERTIES:
        raise ValueError("{} is not a derived property; choose from {}".format(
            name, sorted(DERIVED_PROPERTIES)))
    return float(DERIVED_PROPERTIES[name](mol))


def property_value(example, name):
    """Label of one Phase-1 example: its stored QM9 property when it has one
    of that name, otherwise the derived property computed from its graph."""
    if name in example['properties']:
        return float(example['properties'][name])
    if name not in DERIVED_PROPERTIES:
        raise KeyError("{} is neither a stored QM9 property nor one of the "
                       "derived properties {}".format(
                           name, sorted(DERIVED_PROPERTIES)))
    mol = graph_to_mol(example['atom_types'], example['bond_matrix'])
    return mol_property(mol, name)

