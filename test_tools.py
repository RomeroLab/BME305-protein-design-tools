"""CPU contract tests; no model downloads or GPU required."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import protein_design_tools as tools


def pdb_text():
    lines = []
    serial = 1
    for chain, residues in [('A', ['ALA']), ('B', ['GLY', 'LYS'])]:
        for number, residue in enumerate(residues, 1):
            for atom in ['N', 'CA', 'C', 'O']:
                lines.append(f'ATOM  {serial:5d} {atom:^4s} {residue:3s} {chain}{number:4d}    {float(serial):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{90.:6.2f}          {atom[0]:>2s}\n')
                serial += 1
    lines.append('HETATM   99  O   HOH B  99       0.000   0.000   0.000  1.00 90.00           O\nEND\n')
    return ''.join(lines)


class ToolContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'PROTEIN_DESIGN_CACHE': str(self.root/'cache')})
        self.env.start()
        self.pdb = self.root/'input.pdb'
        self.pdb.write_text(pdb_text())

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_file_chain_selection_and_ligand_removal(self):
        from Bio.PDB import PDBParser
        sequence, path = tools.load_pdb(self.pdb, chain='B', output=self.root/'clean.pdb')
        self.assertEqual(sequence, 'GK')
        model = PDBParser(QUIET=True).get_structure('clean', path)[0]
        self.assertEqual([c.id for c in model], ['B'])
        self.assertEqual(len(list(model.get_residues())), 2)
        self.assertEqual(len(list(model.get_atoms())), 8)

    def test_id_download_is_cached(self):
        def download(url, filename): Path(filename).write_text(pdb_text())
        with patch.object(tools, 'urlretrieve', side_effect=download) as retrieve:
            seq, _ = tools.load_pdb('1ubq', output=self.root/'first.pdb')
            tools.load_pdb('1UBQ', output=self.root/'second.pdb')
        self.assertEqual(seq, 'A')
        self.assertEqual(retrieve.call_count, 1)
        self.assertIn('/1UBQ.pdb', retrieve.call_args.args[0])

    def test_missing_filename_is_not_treated_as_a_pdb_id(self):
        with self.assertRaises(FileNotFoundError): tools.load_pdb(self.root/'missing.pdb')

    def test_mmcif_input(self):
        from Bio.PDB import PDBParser, MMCIFIO
        structure = PDBParser(QUIET=True).get_structure('input', self.pdb)
        writer = MMCIFIO(); writer.set_structure(structure)
        path = self.root/'input.cif'; writer.save(str(path))
        seq, _ = tools.load_pdb(path, chain='B', output=self.root/'from_cif.pdb')
        self.assertEqual(seq, 'GK')

    def test_mpnn_uses_selected_chain_and_returns_only_designs(self):
        _, cleaned = tools.load_pdb(self.pdb, chain='B', output=self.root/'clean.pdb')
        output = self.root/'mpnn'
        def run(args, **kwargs):
            self.assertEqual(args[args.index('--pdb_path_chains')+1], 'B')
            self.assertEqual(args[args.index('--num_seq_per_target')+1], '2')
            (output/'seqs').mkdir(parents=True)
            (output/'seqs'/'clean.fa').write_text('>native\nGK\n>sample1\nAA\n>sample2\nGG\n')
        with patch.object(tools, 'load_MPNN', return_value=self.root/'scripts'), patch.object(tools.subprocess, 'run', side_effect=run):
            designs = tools.MPNNdesign(cleaned, num_sequences=2, output_dir=output)
        self.assertEqual(designs, {'design_1':'AA', 'design_2':'GG'})

    def test_alignment_rejects_silent_truncation(self):
        with self.assertRaises(ValueError): tools.alignment({'a':'AA', 'b':'AAA'})

    def test_esm2_mutation_count_and_repeatability(self):
        import torch
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        class Tokens(dict):
            @property
            def input_ids(self): return self['input_ids']
            def to(self, device): return self
        class Tokenizer:
            mask_token, mask_token_id = '<mask>', 20
            def convert_tokens_to_ids(self, aa): return [amino_acids.index(a) for a in aa]
            def __call__(self, seq, **kwargs):
                letters = seq.replace('<mask>', '?')
                return Tokens(input_ids=torch.tensor([[21]+[20 if a=='?' else amino_acids.index(a) for a in letters]+[22]]))
        def model(**tokens):
            return types.SimpleNamespace(logits=torch.zeros((1, tokens['input_ids'].shape[1], 23)))
        sequence = 'ACDEFGHIKLMNPQRSTVWY'
        with patch.object(tools, 'load_ESM2', return_value=(Tokenizer(), model)):
            first = tools.ESM2design(sequence, n_mutations=5, device='cpu')
            second = tools.ESM2design(sequence, n_mutations=5, device='cpu')
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        for design in first.values(): self.assertEqual(sum(a!=b for a,b in zip(sequence, design)), 5)

    def test_prediction_pdb_confidence_scale_and_independent_outputs(self):
        import torch
        captured = []
        class Tokenizer:
            def __call__(self, seq, **kwargs): return {'input_ids':torch.zeros((1,len(seq)), dtype=torch.long)}
        def model(ids, num_recycles):
            n = ids.shape[1]
            return {'positions':torch.zeros((1,1,n,14,3)), 'aatype':torch.zeros((1,n), dtype=torch.long), 'atom37_atom_exists':torch.ones((1,n,37)), 'residue_index':torch.arange(n)[None,:], 'plddt':torch.full((1,n,37), .9)}
        feats = types.ModuleType('transformers.models.esm.openfold_utils.feats')
        feats.atom14_to_atom37 = lambda positions, folded: torch.zeros((1, positions.shape[1], 37, 3))
        protein = types.ModuleType('transformers.models.esm.openfold_utils.protein')
        protein.Protein = lambda **kwargs: captured.append(kwargs) or kwargs
        protein.to_pdb = lambda obj: 'END\n'
        modules = {feats.__name__: feats, protein.__name__: protein}
        with patch.dict('sys.modules', modules), patch.object(tools, 'load_ESMFold', return_value=(Tokenizer(), model)), patch.object(tools, '_output', side_effect=[self.root/'p1', self.root/'p2']):
            first = tools.predict_structure('ACD', device='cpu')
            second = tools.predict_structure('ACD', device='cpu')
        self.assertNotEqual(first, second)
        self.assertTrue(first.exists() and second.exists())
        self.assertAlmostEqual(float(captured[0]['b_factors'].mean()), 90., places=3)
        self.assertEqual(captured[0]['residue_index'].tolist(), [1,2,3])

    def test_rf_seed_matches_output_filename(self):
        cache = tools._cache()
        (cache/'rf_env/bin').mkdir(parents=True)
        (cache/'rf_env/bin/python').touch()
        (cache/'RFdiffusion/models').mkdir(parents=True)
        (cache/'RFdiffusion/models/Base_ckpt.pt').touch()
        with patch.object(tools.subprocess, 'run') as run:
            result = tools.RFdiffusion(length=80, seed=11, output_dir=self.root/'rf')
        self.assertIn('contigmap.contigs=[80-80]', run.call_args.args[0])
        self.assertIn('inference.design_startnum=11', run.call_args.args[0])
        self.assertEqual(result.name, 'backbone_11.pdb')

    def test_fold_loads_half_before_dispatch_and_preserves_folding_precision(self):
        import torch
        model = torch.nn.Module()
        model.esm = torch.nn.Linear(2,2).half()
        model.trunk = torch.nn.Linear(2,2).half()
        model.trunk.set_chunk_size = lambda size: None
        model.head = torch.nn.Linear(2,2).half()
        model.esm_s_combine = torch.nn.Parameter(torch.zeros(2, dtype=torch.float16))
        module = types.ModuleType('transformers')
        from unittest.mock import Mock
        module.AutoTokenizer = types.SimpleNamespace(from_pretrained=Mock(return_value=object()))
        module.EsmForProteinFolding = types.SimpleNamespace(from_pretrained=Mock(return_value=model))
        with patch.dict('sys.modules', {'transformers':module}), patch.dict(tools._MODELS, {}, clear=True):
            first = tools.load_ESMFold()
            second = tools.load_ESMFold()
        kwargs = module.EsmForProteinFolding.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs['dtype'], torch.float16)
        self.assertEqual(kwargs['device_map'], {'':'cuda'})
        self.assertEqual(model.esm.weight.dtype, torch.float16)
        self.assertEqual(model.trunk.weight.dtype, torch.float32)
        self.assertEqual(model.head.weight.dtype, torch.float32)
        self.assertEqual(model.esm_s_combine.dtype, torch.float32)
        self.assertIs(first, second)
        self.assertEqual(module.EsmForProteinFolding.from_pretrained.call_count, 1)

    def test_saved_bundle_retains_reference_and_prediction(self):
        import zipfile
        bundle = tools.save_results({'design_1':'GK'}, self.pdb, self.pdb, output=self.root/'results.zip')
        with zipfile.ZipFile(bundle) as archive:
            self.assertEqual(set(archive.namelist()), {'designs.fasta','reference.pdb','prediction.pdb'})
            self.assertEqual(archive.read('designs.fasta'), b'>design_1\nGK\n')


if __name__ == '__main__': unittest.main()
