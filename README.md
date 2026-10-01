# BME305 protein design tools

Reusable tools for short Colab demonstrations and later class projects. The notebooks use the same functions students can call on their own sequences and structures.

## Start in Colab

Choose **T4 GPU** and **runtime version 2026.07** (Python 3.12). Run setup before class; model downloads are excluded from the ten-minute walkthrough.

```python
!git -C BME305-protein-design-tools pull -q 2>/dev/null || git clone -q https://github.com/RomeroLab/BME305-protein-design-tools.git
%pip -q install --upgrade ./BME305-protein-design-tools
from protein_design_tools import load_pdb, load_ESMFold, predict_structure, MPNNdesign, alignment, view_structure, inspect_prediction

load_ESMFold()
sequence, backbone = load_pdb('1UBQ', chain='A')
designs = MPNNdesign(backbone, num_sequences=3, temperature=0.1, seed=7)
alignment({'reference': sequence, **designs})
prediction = predict_structure(designs['design_1'], num_recycles=1)
metrics = inspect_prediction(backbone, prediction)
```

Setup clones the tools on first use and pulls updates on later runs. After updating an already imported library, restart the session and rerun setup. PyTorch is supplied by Colab; for another environment, install a suitable PyTorch build first. RFdiffusion needs Linux and a CUDA GPU; the other tools can take `device='cpu'`, but structure prediction will be slow. Dependency versions are pinned to the tested Colab runtime.

## Notebooks

- [ProteinMPNN: ubiquitin](01_ProteinMPNN_ubiquitin.ipynb) · [Open in Colab](https://colab.research.google.com/github/RomeroLab/BME305-protein-design-tools/blob/main/01_ProteinMPNN_ubiquitin.ipynb)
- [ESM-2: sequence generation](02_ESM2_sequence_generation.ipynb) · [Open in Colab](https://colab.research.google.com/github/RomeroLab/BME305-protein-design-tools/blob/main/02_ESM2_sequence_generation.ipynb)
- [RFdiffusion + ProteinMPNN](03_RFdiffusion_ProteinMPNN_denovo.ipynb) · [Open in Colab](https://colab.research.google.com/github/RomeroLab/BME305-protein-design-tools/blob/main/03_RFdiffusion_ProteinMPNN_denovo.ipynb)

## Functions

| Call | Result |
| --- | --- |
| `load_pdb('1UBQ', chain='A')` | `(sequence, cleaned_pdb_path)` |
| `load_pdb('my_structure.pdb', chain='B')` | Same, from a local file; mmCIF also accepted |
| `load_ESMFold()` | Cached tokenizer and prediction model |
| `predict_structure(sequence, num_recycles=1)` | Predicted PDB path |
| `load_MPNN()` | Downloaded scripts/checkpoint folder |
| `MPNNdesign(pdb, num_sequences=3, temperature=0.1, seed=7)` | Dictionary of designed sequences |
| `load_ESM2()` | Cached small ESM-2 tokenizer/model |
| `ESM2design(sequence, num_sequences=3, n_mutations=5, temperature=1, seed=7)` | Dictionary of seed-sequence variants |
| `load_RFdiffusion()` | Isolated RFdiffusion installation and initialized cache |
| `RFdiffusion(length=60, steps=50, seed=7)` | One generated backbone PDB path |
| `alignment(sequences)` | Printed residue comparisons for equal-length designs |
| `view_structure(pdb)` | Interactive structure view |
| `inspect_prediction(reference_pdb, prediction_pdb)` | Overlay and dictionary with `rmsd`, `mean_plddt`, `aligned_pdb` |
| `save_results(designs, reference_pdb, prediction_pdb)` | ZIP path containing FASTA and structures |

`load_pdb` selects the first model and one chain, removes water and ligands, and keeps standard amino acids with complete backbone atoms. If `chain` is omitted it selects the first protein chain. Returned paths are `pathlib.Path` objects; pass them directly to the other functions. Optional `output` or `output_dir` arguments let you choose destinations. Otherwise each run gets its own directory under `protein_demo/`, so later runs do not overwrite earlier results. Set `PROTEIN_DESIGN_CACHE` to change the download cache.

The wrappers support single-chain design, seeded ESM-2 variants, and unconditional RFdiffusion backbones. They do not implement multi-chain design, motif scaffolding, or binders. Alignments assume equal lengths and matching residue positions; structure overlays assume matching Cα correspondence. Use a gapped alignment/structural alignment tool for unrelated proteins or insertions/deletions.

ESM-2 is a masked language model, so `ESM2design` generates variants around a supplied sequence. ESMFold takes the sequence alone and writes pLDDT on a 0–100 scale into PDB B factors. A matching predicted fold and high confidence do not establish experimental folding or function. Large proteins require more time and memory than the 60–76-residue demos.

## Implementation and validation

The upstream model revisions and dependencies retain the versions used in the original demos. Those demos ran on a free Colab T4; their design/inspection runs were below ten minutes after setup. This shared-library refactor has CPU contract tests; its GPU pathways should receive a fresh classroom trial before teaching. Run `python -m unittest -v test_tools` from the repository to check file loading and wrapper contracts without downloading models.

Model weights and upstream software are downloaded during setup, not included here. Their original licenses and terms apply: [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), [ESM-2 / ESMFold](https://github.com/facebookresearch/esm), [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion), and the [RFdiffusion Colab adaptation](https://github.com/sokrypton/RFdiffusion).

### ESMFold loading memory

Version 0.1.1 loads CUDA weights directly in float16, then restores float32 only for the smaller folding trunk, heads, and layer-combination weights. This avoids the initial full-float GPU allocation that can exhaust a T4 before the old loader reaches its precision conversion. A regression test checks loading dtype, mixed precision and reuse of the cached model. After a CUDA out-of-memory error, restart the session before rerunning the updated setup to release tensors retained by the failed call.
