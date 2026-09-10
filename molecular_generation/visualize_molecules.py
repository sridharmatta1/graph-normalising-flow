"""Phase 6 add-on: renders a grid image of actual generated molecule
structures from generate_molecules.py's --save_results_json output, for
presenting results (e.g. to a professor) as pictures of real molecules
rather than just validity/uniqueness/novelty percentages.

Deliberately a separate script from generate_molecules.py rather than
folding drawing into it -- generation needs both TF checkpoints restored
(slow, GPU), drawing only needs the saved SMILES (fast, CPU-only, no TF
at all) -- keeping them separate means re-rendering with different
grid/filtering options never requires re-running generation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

flags.DEFINE_string(
    'results_json',
    'molecular_generation/results/generated_molecules.json',
    'Output of generate_molecules.py --save_results_json.')
flags.DEFINE_string(
    'output_image',
    'molecular_generation/results/generated_molecules_grid.png', '')
flags.DEFINE_integer('max_molecules', 20,
                     'How many distinct molecules to draw.')
flags.DEFINE_integer('mols_per_row', 5, '')
flags.DEFINE_integer('sub_img_size', 260,
                     'Pixel size (square) of each individual molecule '
                     'cell in the grid.')
flags.DEFINE_bool(
    'only_novel', False,
    'If set, only draw molecules flagged novel (not present in the '
    'QM9 training set) -- the more interesting case for showing the '
    'model actually generalizes rather than memorizes.')
FLAGS = flags.FLAGS


def main(argv):
    del argv
    with open(FLAGS.results_json, 'r') as f:
        payload = json.load(f)

    summary = payload['summary']
    molecules = payload['molecules']

    # Strictly-valid (single connected molecule) only -- a fragmented
    # output isn't a real molecule and would just draw as disconnected
    # blobs, which is confusing rather than informative in a figure.
    connected = [m for m in molecules if m['connected'] and m['smiles']]
    if FLAGS.only_novel:
        connected = [m for m in connected if m['novel']]

    seen = set()
    deduped = []
    for m in connected:
        if m['smiles'] not in seen:
            seen.add(m['smiles'])
            deduped.append(m)
    deduped = deduped[:FLAGS.max_molecules]

    print("Summary metrics from generation run: {}".format(summary))
    print("Drawing {} distinct{} molecules out of {} available.".format(
        len(deduped), " novel" if FLAGS.only_novel else "", len(connected)))

    mols = []
    legends = []
    for i, m in enumerate(deduped):
        mol = Chem.MolFromSmiles(m['smiles'])
        if mol is None:
            continue
        mols.append(mol)
        tag = "novel" if m['novel'] else "seen in training set"
        legends.append("{}. {} ({})".format(i + 1, m['smiles'], tag))

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=FLAGS.mols_per_row,
        subImgSize=(FLAGS.sub_img_size, FLAGS.sub_img_size),
        legends=legends,
        useSVG=False)

    os.makedirs(os.path.dirname(FLAGS.output_image) or '.', exist_ok=True)
    # MolsToGridImage returns a PIL Image when useSVG=False.
    img.save(FLAGS.output_image)
    print("Saved grid image of {} molecules to {}".format(
        len(mols), FLAGS.output_image))


if __name__ == '__main__':
    app.run(main)
