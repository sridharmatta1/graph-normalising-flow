"""Shared QM9 constants, kept separate from preprocess_qm9.py so other
modules (qm9_graph_data.py) can import them without also triggering that
script's absl.flags.DEFINE_* calls -- those register into a single
process-wide flag registry, and collide (DuplicateFlagError) with any
other script in the same process that defines a flag of the same name
(e.g. max_molecules, random_seed, both defined in
train_molecular_autoencoder.py too).
"""

# QM9 (v1) molecules only ever contain these 4 heavy elements (plus H,
# which we don't model as explicit nodes -- standard convention for
# graph-based molecule generation, since H count is recoverable from
# valence).
ATOM_VOCAB = ['C', 'N', 'O', 'F']
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_VOCAB)}
