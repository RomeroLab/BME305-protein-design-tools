"""Small, reusable protein-design tools for Colab teaching notebooks."""
from pathlib import Path
from urllib.request import urlretrieve
import os
import random
import re
import subprocess
import sys
import tempfile

__all__ = ['load_pdb', 'load_ESMFold', 'load_ESM2', 'load_MPNN', 'load_RFdiffusion', 'predict_structure', 'MPNNdesign', 'ESM2design', 'RFdiffusion', 'alignment', 'view_structure', 'inspect_prediction', 'save_results']
_MODELS = {}


def _cache():
    path = Path(os.environ.get('PROTEIN_DESIGN_CACHE', 'protein_demo/cache')).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output(prefix):
    root = Path('protein_demo').resolve()
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix+'_', dir=root))


def load_pdb(source, chain=None, output=None):
    """Return (sequence, cleaned_pdb_path) from a PDB ID or local PDB/mmCIF file.

    Select the first model and one chain (first protein chain by default).
    Keep standard amino acids with complete N/CA/C/O backbone atoms; remove
    water, ligands and incomplete residues. Preserve residue numbers and chain ID.
    """
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
    from Bio.SeqUtils import seq1
    from Bio.PDB.Polypeptide import is_aa
    source = str(source)
    path = Path(source).expanduser()
    if not path.is_file():
        if not re.fullmatch(r'[0-9][A-Za-z0-9]{3}', source):
            raise FileNotFoundError(source)
        path = _cache() / (source.upper()+'.pdb')
        if not path.exists():
            urlretrieve(f'https://files.rcsb.org/download/{source.upper()}.pdb', path)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in ('.cif', '.mmcif') else PDBParser(QUIET=True)
    model = parser.get_structure('input', path)[0]
    if chain is None:
        chain = next(c.id for c in model if any(is_aa(r, standard=True) for r in c))
    selected = model[chain]
    residues = [r for r in selected if is_aa(r, standard=True) and all(a in r for a in ('N', 'CA', 'C', 'O'))]
    if not residues:
        raise ValueError(f'Chain {chain} has no standard residues with complete backbone atoms.')
    if len(chain) != 1:
        raise ValueError('PDB output requires a one-character chain ID.')
    residue_ids = {r.id for r in residues}
    class ProteinChain(Select):
        def accept_chain(self, c): return c.id == chain
        def accept_residue(self, r): return r.id in residue_ids
    output = Path(output) if output else _output('input') / (path.stem+'_'+chain+'.pdb')
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PDBIO(); writer.set_structure(model); writer.save(str(output), ProteinChain())
    sequence = ''.join(seq1(r.resname) for r in residues)
    print(f'{path.stem}, chain {chain}: {len(sequence)} residues')
    return sequence, output


def load_ESMFold(device='cuda'):
    """Download/cache ESMFold once, with memory settings for a free Colab GPU."""
    from transformers import AutoTokenizer, EsmForProteinFolding
    key = ('ESMFold', device)
    if key not in _MODELS:
        revision = '75a3841ee059df2bf4d56688166c8fb459ddd97a'
        tokenizer = AutoTokenizer.from_pretrained('facebook/esmfold_v1', revision=revision)
        model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1', revision=revision, device_map={'':device}).eval()
        if device.startswith('cuda'): model.esm = model.esm.half()
        model.trunk.set_chunk_size(64)
        _MODELS[key] = tokenizer, model
    return _MODELS[key]


def predict_structure(sequence, output=None, num_recycles=1, device='cuda'):
    """Predict one sequence; return a PDB path whose B factors hold pLDDT (0–100)."""
    import torch
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    from transformers.models.esm.openfold_utils.protein import Protein, to_pdb
    tokenizer, model = load_ESMFold(device)
    sequence = ''.join(sequence.split()).upper()
    ids = tokenizer(sequence, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    with torch.inference_mode():
        folded = model(ids, num_recycles=num_recycles)
        positions = atom14_to_atom37(folded['positions'][-1], folded).cpu().numpy()[0]
        folded = {k:v.cpu().numpy() for k,v in folded.items()}
    output = Path(output) if output else _output('prediction') / 'prediction.pdb'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(to_pdb(Protein(aatype=folded['aatype'][0], atom_positions=positions, atom_mask=folded['atom37_atom_exists'][0], residue_index=folded['residue_index'][0]+1, b_factors=100*folded['plddt'][0])))
    del folded, positions, ids
    if device.startswith('cuda'): torch.cuda.empty_cache()
    print(f'Predicted {len(sequence)} residues: {output}')
    return output


def load_MPNN():
    """Download the pinned ProteinMPNN scripts and v_48_020 checkpoint."""
    folder = _cache() / 'ProteinMPNN'; folder.mkdir(exist_ok=True)
    revision = '8907e6671bfbfc92303b5f79c4b5e6ce47cdef57'
    for filename in ['protein_mpnn_run.py', 'protein_mpnn_utils.py', 'vanilla_model_weights/v_48_020.pt']:
        dest = folder / filename; dest.parent.mkdir(exist_ok=True)
        if not dest.exists(): urlretrieve(f'https://raw.githubusercontent.com/dauparas/ProteinMPNN/{revision}/'+filename, dest)
    script = folder / 'protein_mpnn_run.py'
    script.write_text(script.read_text().replace('torch.load(checkpoint_path, map_location=device)', 'torch.load(checkpoint_path, map_location=device, weights_only=False)'))
    return folder


def MPNNdesign(pdb, num_sequences=3, temperature=0.1, seed=7, chain=None, output_dir=None):
    """Design one protein chain; return {design_1: sequence, ...}.

    Input must be a cleaned single-chain PDB (e.g. the result of load_pdb).
    Files and MPNN scores are retained in output_dir, or a new results directory.
    """
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    pdb = Path(pdb).resolve()
    chains = list(PDBParser(QUIET=True).get_structure('input', pdb)[0])
    if len(chains) != 1:
        raise ValueError('Use load_pdb(source, chain=...) to select one chain first.')
    chain = chain or chains[0].id
    folder = load_MPNN()
    output_dir = Path(output_dir).resolve() if output_dir else _output('mpnn')
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(folder/'protein_mpnn_run.py'), '--pdb_path', str(pdb), '--pdb_path_chains', chain, '--out_folder', str(output_dir), '--num_seq_per_target', str(num_sequences), '--sampling_temp', str(temperature), '--seed', str(seed), '--batch_size', '1'], check=True)
    with (output_dir/'seqs'/(pdb.stem+'.fa')).open() as fasta:
        records = list(SeqIO.parse(fasta, 'fasta'))
    return {f'design_{i}':str(record.seq) for i,record in enumerate(records[1:], 1)}


def load_ESM2(device='cuda'):
    """Download/cache the small 8M-parameter ESM-2 masked language model."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    key = ('ESM2', device)
    if key not in _MODELS:
        name = 'facebook/esm2_t6_8M_ur50d'
        revision = 'c731040fcd8d73dceaa04b0a8e6329b345b0f5df'
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForMaskedLM.from_pretrained(name, revision=revision).eval().to(device)
        _MODELS[key] = tokenizer, model
    return _MODELS[key]


def ESM2design(sequence, num_sequences=3, n_mutations=5, temperature=1, seed=7, device='cuda'):
    """Generate seed-sequence variants with exactly n_mutations distinct substitutions."""
    import torch
    tokenizer, model = load_ESM2(device)
    sequence = ''.join(sequence.split()).upper()
    if not 0 <= n_mutations <= len(sequence) or temperature <= 0:
        raise ValueError('Use 0 ≤ n_mutations ≤ sequence length and temperature > 0.')
    rng = random.Random(seed); torch.manual_seed(seed)
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    aa_ids = tokenizer.convert_tokens_to_ids(amino_acids)
    designs = {}
    for n in range(1, num_sequences+1):
        seq = sequence
        for pos in rng.sample(range(len(seq)), n_mutations):
            masked = seq[:pos]+tokenizer.mask_token+seq[pos+1:]
            tokens = tokenizer(masked, return_tensors='pt').to(device)
            mask_pos = (tokens.input_ids[0]==tokenizer.mask_token_id).nonzero()[0,0]
            with torch.no_grad(): logits = model(**tokens).logits[0, mask_pos, aa_ids]
            logits[amino_acids.index(sequence[pos])] = -torch.inf
            probs = torch.softmax(logits/temperature, dim=-1)
            aa = amino_acids[torch.multinomial(probs, 1).item()]
            seq = seq[:pos]+aa+seq[pos+1:]
        designs[f'design_{n}'] = seq
    print(f'Generated {num_sequences} sequences with {n_mutations} substitutions each.')
    return designs


def load_RFdiffusion():
    """Install RFdiffusion in an isolated environment and initialize its diffusion cache."""
    work = _cache()
    # Use a separate environment for RFdiffusion's older PyTorch/DGL combination.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.22'], check=True)
    rf = work / 'RFdiffusion'
    if not rf.exists():
        subprocess.run(['git','clone','-q','https://github.com/sokrypton/RFdiffusion.git',str(rf)],check=True)
        subprocess.run(['git','-C',str(rf),'checkout','-q','597d37f2a686e23941440fddf6daa4cb778e7bc7'],check=True)
    rf_python = work / 'rf_env/bin/python'
    if not rf_python.exists():
        subprocess.run([sys.executable, '-m', 'uv','venv','--python',sys.executable,str(work/'rf_env')],check=True)
    if not (work/'rf_setup.done').exists():
        subprocess.run([sys.executable, '-m', 'uv','pip','install','-q','--python',str(rf_python),'torch==2.4.1+cu124', '--extra-index-url','https://download.pytorch.org/whl/cu124', '--index-strategy','unsafe-best-match'],check=True)
        subprocess.run([sys.executable, '-m', 'uv','pip','install','-q','--python',str(rf_python),'numpy==1.26.4','scipy==1.13.1', 'dgl==2.4.0+cu124','-f','https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html', 'e3nn==0.5.5','opt_einsum_fx','hydra-core==1.3.2','icecream','pyrsistent','pynvml','decorator', 'git+https://github.com/NVIDIA/dllogger',str(rf/'env/SE3Transformer')],check=True)
        (work/'rf_setup.done').touch()
    (rf/'models').mkdir(exist_ok=True)
    if not (rf/'models/Base_ckpt.pt').exists():
        urlretrieve('https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt', rf/'models/Base_ckpt.pt')
    # Initialize the diffusion cache before class, without generating a design.
    subprocess.run([str(rf_python),str(rf/'run_inference.py'), 'contigmap.contigs=[60-60]','diffuser.T=50','inference.num_designs=0', f'inference.ckpt_override_path={rf}/models/Base_ckpt.pt'], env=dict(os.environ,DGLBACKEND='pytorch'),check=True)
    print('RFdiffusion ready.')
    return rf, rf_python


def RFdiffusion(length=60, steps=50, seed=7, output_dir=None):
    """Generate one unconditional backbone and return its PDB path.

    This short classroom wrapper supports length-only generation, not binders
    or motif scaffolding. Run load_RFdiffusion before class to finish setup.
    """
    rf, rf_python = _cache()/'RFdiffusion', _cache()/'rf_env/bin/python'
    if not rf_python.exists() or not (rf/'models/Base_ckpt.pt').exists():
        rf, rf_python = load_RFdiffusion()
    output_dir = Path(output_dir).resolve() if output_dir else _output('rf')
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / 'backbone'
    subprocess.run([str(rf_python), str(rf/'run_inference.py'), f'contigmap.contigs=[{length}-{length}]', f'diffuser.T={steps}', 'inference.num_designs=1', f'inference.design_startnum={seed}', 'inference.deterministic=True', f'inference.output_prefix={prefix}', f'inference.ckpt_override_path={rf}/models/Base_ckpt.pt'], env=dict(os.environ, DGLBACKEND='pytorch'), check=True)
    return Path(str(prefix)+f'_{seed}.pdb')

def alignment(sequences):
    """Display an ungapped alignment of equal-length, corresponding sequences."""
    if len({len(s) for s in sequences.values()}) > 1:
        raise ValueError("This alignment compares equal-length designs; use a gapped aligner for insertions/deletions.")
    # These designs have the same length and residue correspondence: no gaps needed.
    names = list(sequences); reference = sequences[names[0]]
    for start in range(0, len(reference), 60):
        end = min(start+60, len(reference))
        print(f'\nPositions {start+1}–{end}')
        for name, sequence in sequences.items():
            print(f'{name:12s} {sequence[start:end]}')
            if name != names[0]:
                print(' '*13 + ''.join('|' if a == b else '.' for a,b in zip(reference[start:end],sequence[start:end])))
    for name, sequence in list(sequences.items())[1:]:
        changes = [f'{a}{i+1}{b}' for i,(a,b) in enumerate(zip(reference,sequence)) if a!=b]
        identity = sum(a==b for a,b in zip(reference,sequence))/len(reference)
        print(f'\n{name}: {identity:.0%} identity to {names[0]}; {len(changes)} differences')
        print(', '.join(changes))

def view_structure(filename):
    """Display a PDB file with py3Dmol."""
    import py3Dmol
    view = py3Dmol.view(width=700, height=400)
    view.addModel(Path(filename).read_text(), 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo(); view.show()

def inspect_prediction(reference_path, prediction_path):
    """Overlay corresponding Cα atoms; report RMSD and prediction confidence."""
    import numpy as np
    import py3Dmol
    from Bio.PDB import PDBParser, PDBIO, Superimposer
    parser = PDBParser(QUIET=True)
    ref = parser.get_structure('reference', reference_path)[0]
    pred = parser.get_structure('prediction', prediction_path)[0]
    ref_ca = [r['CA'] for r in ref.get_residues() if r.id[0]==' ' and 'CA' in r]
    pred_ca = [r['CA'] for r in pred.get_residues() if r.id[0]==' ' and 'CA' in r]
    if len(ref_ca) != len(pred_ca):
        raise ValueError('Reference and prediction need matching residue correspondence and counts.')
    sup = Superimposer(); sup.set_atoms(ref_ca,pred_ca); sup.apply(list(pred.get_atoms()))
    output = Path(prediction_path).with_name('prediction_aligned.pdb')
    io = PDBIO(); io.set_structure(pred); io.save(str(output))
    print(f'Cα RMSD over all {len(ref_ca)} residues: {sup.rms:.2f} Å')
    print(f'Mean Cα pLDDT: {np.mean([a.bfactor for a in pred_ca]):.1f}/100')
    view = py3Dmol.view(width=700,height=430)
    view.addModel(Path(reference_path).read_text(),'pdb')
    view.addModel(output.read_text(),'pdb')
    view.setStyle({'model':0},{'cartoon':{'color':'lightgray','opacity':0.6}})
    view.setStyle({'model':1},{'cartoon':{'color':'blue'}})
    for atom in pred_ca:
        color = 'blue' if atom.bfactor>=90 else 'cyan' if atom.bfactor>=70 else 'yellow' if atom.bfactor>=50 else 'red'
        view.setStyle({'model':1,'chain':atom.parent.parent.id,'resi':atom.parent.id[1]}, {'cartoon':{'color':color}})
    view.zoomTo(); view.show()
    print('Gray: reference. Prediction confidence: blue ≥90, cyan 70–90, yellow 50–70, red <50.')
    return {'rmsd':float(sup.rms), 'mean_plddt':float(np.mean([a.bfactor for a in pred_ca])), 'aligned_pdb':output}


def save_results(designs, reference_pdb, prediction_pdb, output='protein_design_results.zip'):
    """Save FASTA and structures to a ZIP; return its path (no automatic download)."""
    import zipfile
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fasta = ''.join(f'>{name}\n{seq}\n' for name,seq in designs.items())
    with zipfile.ZipFile(output, 'w') as archive:
        archive.writestr('designs.fasta', fasta)
        archive.write(reference_pdb, 'reference.pdb')
        archive.write(prediction_pdb, 'prediction.pdb')
        aligned = Path(prediction_pdb).with_name('prediction_aligned.pdb')
        if aligned.exists(): archive.write(aligned, 'prediction_aligned.pdb')
    return output
