# BME305 protein design tools

Explore three approaches to protein design: propose sequences for a known backbone, generate variants from sequence context, and build a new protein from a generated backbone. Each notebook connects design to sequence comparison and structure prediction.

The shared Python tools also accept your own sequences and structures for later assignments.

## Notebooks

| Notebook | Input | Design objective | Open |
| --- | --- | --- | --- |
| [ProteinMPNN](01_ProteinMPNN_ubiquitin.ipynb) | Ubiquitin backbone | Alternative sequences compatible with a fixed backbone | [Colab](https://colab.research.google.com/github/RomeroLab/BME305-protein-design-tools/blob/main/01_ProteinMPNN_ubiquitin.ipynb) |
| [ESM-2](02_ESM2_sequence_generation.ipynb) | Ubiquitin sequence | Variants with a specified number of substitutions, sampled from sequence context | [Colab](https://colab.research.google.com/github/RomeroLab/BME305-protein-design-tools/blob/main/02_ESM2_sequence_generation.ipynb) |
| [RFdiffusion + ProteinMPNN](03_RFdiffusion_ProteinMPNN_denovo.ipynb) | Target length | A new backbone and sequences compatible with it | [Colab](https://colab.research.google.com/github/RomeroLab/BME305-protein-design-tools/blob/main/03_RFdiffusion_ProteinMPNN_denovo.ipynb) |

## Getting started

Open a notebook, select **Runtime → Change runtime type → T4 GPU**, and run its setup cells. Then work through design, sequence comparison, and structure inspection.

To use the tools in another Colab notebook:

```python
!git -C BME305-protein-design-tools pull -q 2>/dev/null || git clone -q https://github.com/RomeroLab/BME305-protein-design-tools.git
%pip -q install --upgrade ./BME305-protein-design-tools
from protein_design_tools import load_pdb, load_ESMFold, MPNNdesign, predict_structure, alignment, inspect_prediction
```

```python
load_ESMFold();
sequence, backbone = load_pdb('1UBQ', chain='A')
designs = MPNNdesign(backbone, num_sequences=3, temperature=0.1, seed=7)
alignment({'reference': sequence, **designs})
prediction = predict_structure(designs['design_1'], num_recycles=1)
metrics = inspect_prediction(backbone, prediction)
```

For your own structure, replace the PDB ID with a filename: `load_pdb('my_structure.pdb', chain='B')`. To predict a sequence directly, use `predict_structure(sequence)`.

## Function reference

| Function | Purpose and return value |
| --- | --- |
| `load_pdb(source, chain=None)` | Load a PDB ID or local PDB/mmCIF file; return `(sequence, cleaned_pdb_path)`. |
| `load_ESMFold()` | Load and cache the structure predictor. |
| `predict_structure(sequence, num_recycles=1)` | Predict a structure; return its PDB path. |
| `load_MPNN()` | Download ProteinMPNN scripts and weights. |
| `MPNNdesign(pdb, num_sequences=3, temperature=0.1, seed=7)` | Design sequences for a backbone; return a dictionary of sequences. |
| `load_ESM2()` | Load and cache the ESM-2 sequence model. |
| `ESM2design(sequence, num_sequences=3, n_mutations=5, temperature=1, seed=7)` | Generate variants with a specified number of substitutions; return a dictionary of sequences. |
| `load_RFdiffusion()` | Initialize RFdiffusion and its dependencies. |
| `RFdiffusion(length=60, steps=50, seed=7)` | Generate a backbone; return its PDB path. |
| `alignment(sequences)` | Display residue comparisons for corresponding sequences. |
| `view_structure(pdb)` | Display an interactive structure view. |
| `inspect_prediction(reference_pdb, prediction_pdb)` | Overlay structures; return `rmsd`, `mean_plddt`, and `aligned_pdb`. |
| `save_results(designs, reference_pdb, prediction_pdb)` | Save sequences and structures; return a ZIP path. |

`load_pdb` selects one chain from the first model, removes water and ligands, and keeps standard amino acids with complete backbone atoms. If `chain` is omitted, it selects the first protein chain. Returned paths can be passed directly between functions.

Optional `output` and `output_dir` arguments let you choose where results are saved. Otherwise, each run creates a new directory under `protein_demo/`. Downloads are cached; set `PROTEIN_DESIGN_CACHE` to choose a cache location.

## Interpreting designs

ProteinMPNN conditions sequence generation on a backbone. ESM-2 proposes variants from sequence context. RFdiffusion generates a backbone that can then be assigned a sequence with ProteinMPNN. ESMFold predicts from sequence alone, without the reference coordinates, providing a separate comparison with the reference backbone.

RMSD measures backbone agreement; pLDDT describes confidence in the prediction. A matching predicted fold and high confidence do not establish experimental folding or biological function.

These tools support single-chain design, ESM-2 variants around a seed sequence, and unconditional RFdiffusion backbones. Sequence comparisons assume equal lengths and corresponding positions; structure overlays assume corresponding Cα atoms. For insertions, deletions, or unrelated proteins, use an appropriate sequence or structural alignment method.

## Development

Dependency versions and upstream model revisions are pinned for reproducibility. Colab provides PyTorch; other environments need a suitable PyTorch installation. RFdiffusion uses an isolated environment and requires Linux with a CUDA GPU. ESM-2 and ESMFold also accept `device='cpu'`.

Run `python -m unittest -v test_tools` to check file loading, sampling, model loading, and result formats without downloading models. The ProteinMPNN and ESMFold workflow has also been verified on a Colab T4. After updating an imported library, restart the session and rerun setup.

## Sources

Model weights and upstream software are downloaded during setup. Their original licenses and terms apply: [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), [ESM-2 and ESMFold](https://github.com/facebookresearch/esm), [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion), and the [RFdiffusion Colab adaptation](https://github.com/sokrypton/RFdiffusion).
