"""
Dataset and graph utilities for molecular property prediction.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


ATOM_SYMBOLS = [
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "As",
    "Se",
    "Br",
    "Te",
    "I",
    "At",
    "other",
]
ATOM_DEGREES = [0, 1, 2, 3, 4, 5, "other"]
ATOM_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
]
BOND_STEREO_TYPES = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
HALOGEN_SYMBOLS = {"F", "Cl", "Br", "I", "At"}
PHARMACO_PROPS = ["Hbond_donor", "Hbond_acceptor", "Basic", "Acid", "Halogen"]
DEFAULT_EXPLICIT_H = True
DEFAULT_USE_CHIRALITY = True
DEFAULT_USE_PHARMACO = True
DEFAULT_USE_SCAFFOLD = True

# Atom feature component dimensions
ATOM_CHARGE_RADICAL_DIM = 2
ATOM_AROMATIC_DIM = 1
ATOM_TOTAL_H_DIM = 5  # one-hot over [0, 1, 2, 3, 4]
ATOM_CHIRALITY_DIM = 3  # one-hot(R/S) + ChiralityPossible
ATOM_PHARMACO_DIM = len(PHARMACO_PROPS)
ATOM_SCAFFOLD_DIM = 1

# Bond feature component dimensions
BOND_TYPE_DIM = 4  # single/double/triple/aromatic
BOND_CONJUGATED_DIM = 1
BOND_IN_RING_DIM = 1

ATOM_FEATURE_DIM = (
    len(ATOM_SYMBOLS)
    + len(ATOM_DEGREES)
    + ATOM_CHARGE_RADICAL_DIM
    + len(ATOM_HYBRIDIZATIONS)
    + ATOM_AROMATIC_DIM
    + (0 if DEFAULT_EXPLICIT_H else ATOM_TOTAL_H_DIM)
    + (ATOM_CHIRALITY_DIM if DEFAULT_USE_CHIRALITY else 0)
    + (ATOM_PHARMACO_DIM if DEFAULT_USE_PHARMACO else 0)
    + (ATOM_SCAFFOLD_DIM if DEFAULT_USE_SCAFFOLD else 0)
)
BOND_FEATURE_DIM = BOND_TYPE_DIM + BOND_CONJUGATED_DIM + BOND_IN_RING_DIM + len(BOND_STEREO_TYPES)

FEATURE_FACTORY = None
try:
    FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )
except Exception:
    FEATURE_FACTORY = None


def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}:")
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def tag_pharmacophore(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    for atom in mol.GetAtoms():
        for prop in PHARMACO_PROPS:
            atom.SetProp(prop, "0")

    if FEATURE_FACTORY is not None:
        for feat in FEATURE_FACTORY.GetFeaturesForMol(mol):
            fam = feat.GetFamily()
            if fam == "Donor":
                prop = "Hbond_donor"
            elif fam == "Acceptor":
                prop = "Hbond_acceptor"
            elif fam == "PosIonizable":
                prop = "Basic"
            elif fam == "NegIonizable":
                prop = "Acid"
            else:
                continue
            for atom_id in feat.GetAtomIds():
                mol.GetAtomWithIdx(atom_id).SetProp(prop, "1")

    for atom in mol.GetAtoms():
        if atom.GetSymbol() in HALOGEN_SYMBOLS:
            atom.SetProp("Halogen", "1")

    return mol


def tag_scaffold(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    for atom in mol.GetAtoms():
        atom.SetProp("Scaffold", "0")

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return mol
        for match in mol.GetSubstructMatches(scaffold):
            for atom_id in match:
                mol.GetAtomWithIdx(atom_id).SetProp("Scaffold", "1")
    except Exception:
        return mol

    return mol


def atom_attr(
    mol: Chem.rdchem.Mol,
    explicit_H: bool = DEFAULT_EXPLICIT_H,
    use_chirality: bool = DEFAULT_USE_CHIRALITY,
    pharmaco: bool = DEFAULT_USE_PHARMACO,
    scaffold: bool = DEFAULT_USE_SCAFFOLD,
) -> np.ndarray:
    if pharmaco:
        mol = tag_pharmacophore(mol)
    if scaffold:
        mol = tag_scaffold(mol)

    feat = []
    for atom in mol.GetAtoms():
        results = (
            onehot_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS)
            + onehot_encoding_unk(atom.GetDegree(), ATOM_DEGREES)
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            + onehot_encoding_unk(atom.GetHybridization(), ATOM_HYBRIDIZATIONS)
            + [atom.GetIsAromatic()]
        )
        if not explicit_H:
            results = results + onehot_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = (
                    results
                    + onehot_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"])
                    + [atom.HasProp("_ChiralityPossible")]
                )
            except Exception:
                results = results + [0, 0] + [atom.HasProp("_ChiralityPossible")]
        if pharmaco:
            results = (
                results
                + [int(atom.GetProp("Hbond_donor"))]
                + [int(atom.GetProp("Hbond_acceptor"))]
                + [int(atom.GetProp("Basic"))]
                + [int(atom.GetProp("Acid"))]
                + [int(atom.GetProp("Halogen"))]
            )
        if scaffold:
            results = results + [int(atom.GetProp("Scaffold"))]
        feat.append([float(v) for v in results])

    if len(feat) == 0:
        return np.empty((0, ATOM_FEATURE_DIM), dtype=np.float32)
    return np.asarray(feat, dtype=np.float32)


def bond_attr(mol: Chem.rdchem.Mol, use_chirality: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is None:
                continue
            bt = bond.GetBondType()
            bond_feats = [
                bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                bond.GetIsConjugated(),
                bond.IsInRing(),
            ]
            if use_chirality:
                bond_feats = bond_feats + onehot_encoding_unk(
                    str(bond.GetStereo()), BOND_STEREO_TYPES
                )
            feat.append([float(v) for v in bond_feats])
            index.append([i, j])

    if len(index) == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, BOND_FEATURE_DIM), dtype=np.float32),
        )
    return np.asarray(index, dtype=np.int64), np.asarray(feat, dtype=np.float32)


def smiles_to_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = atom_attr(
        mol,
        explicit_H=DEFAULT_EXPLICIT_H,
        use_chirality=DEFAULT_USE_CHIRALITY,
        pharmaco=DEFAULT_USE_PHARMACO,
        scaffold=DEFAULT_USE_SCAFFOLD,
    )
    if atom_features.shape[0] == 0:
        return None

    edge_index_arr, edge_attr_arr = bond_attr(mol)

    if edge_index_arr.shape[0] == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
    else:
        edge_index = torch.from_numpy(edge_index_arr.T).to(torch.long).contiguous()
        edge_attr = torch.from_numpy(edge_attr_arr).to(torch.float32)

    x = torch.from_numpy(atom_features).to(torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


DATASET_CONFIG: Dict[str, Dict[str, object]] = {
    "HIV": {"task_type": "classification", "label_cols": ["Class"]},
    "BACE": {"task_type": "classification", "label_cols": ["Class"]},
    "BBBP": {"task_type": "classification", "label_cols": ["Class"]},
    "ClinTox": {"task_type": "multilabel", "label_cols": ["FDA_APPROVED", "CT_TOX"]},
    "Tox21": {
        "task_type": "multilabel",
        "label_cols": [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
    },
    "ESOL": {"task_type": "regression", "label_cols": ["logSolubility"]},
    "FreeSolv": {"task_type": "regression", "label_cols": ["freesolv"]},
    "Lipophilicity": {"task_type": "regression", "label_cols": ["lipo"]},
    "SIDER": {
        "task_type": "multilabel",
        "label_cols": [f"SIDER{i}" for i in range(1, 28)],
    },
}


class MolecularDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "dataset",
        embeddings_dir: str = "embeddings",
        task_type: Optional[str] = None,
    ):
        super().__init__()
        if dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.dataset_name = dataset_name
        cfg = DATASET_CONFIG[dataset_name]
        self.task_type = task_type or str(cfg["task_type"])
        self.label_cols = list(cfg["label_cols"])

        csv_path = os.path.join(data_dir, f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)

        emb_path = os.path.join(embeddings_dir, f"{dataset_name}_embeddings.npy")
        self.precomputed_embeddings = None
        if os.path.exists(emb_path):
            self.precomputed_embeddings = np.load(emb_path)
            if len(self.precomputed_embeddings) != len(df):
                raise ValueError(
                    f"Embedding rows ({len(self.precomputed_embeddings)}) do not match CSV rows ({len(df)}) for {dataset_name}"
                )

        self.samples = []
        self.valid_labels = []
        dropped_invalid = 0
        dropped_missing_label = 0

        for idx, row in df.iterrows():
            smiles = row.get("smiles")
            if not isinstance(smiles, str) or len(smiles.strip()) == 0:
                dropped_invalid += 1
                continue

            graph = smiles_to_graph(smiles)
            if graph is None:
                dropped_invalid += 1
                continue

            if self.task_type == "multilabel":
                labels = row[self.label_cols].fillna(0).to_numpy(dtype=np.float32)
                label_tensor = torch.tensor(labels, dtype=torch.float32)
            elif self.task_type == "regression":
                val = row[self.label_cols[0]]
                if pd.isna(val):
                    dropped_missing_label += 1
                    continue
                label_tensor = torch.tensor(float(val), dtype=torch.float32)
            else:
                val = row[self.label_cols[0]]
                if pd.isna(val):
                    dropped_missing_label += 1
                    continue
                label_tensor = torch.tensor(int(val), dtype=torch.long)
                self.valid_labels.append(int(val))

            smiles_embedding = None
            if self.precomputed_embeddings is not None:
                smiles_embedding = torch.tensor(
                    self.precomputed_embeddings[idx], dtype=torch.float32
                )

            self.samples.append(
                {
                    "graph": graph,
                    "smiles": smiles,
                    "label": label_tensor,
                    "smiles_embedding": smiles_embedding,
                }
            )

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples loaded for dataset {dataset_name}")

        print(
            f"[{dataset_name}] loaded {len(self.samples)} samples "
            f"(dropped_invalid={dropped_invalid}, dropped_missing_label={dropped_missing_label})"
        )
        if self.precomputed_embeddings is not None:
            print(f"[{dataset_name}] using precomputed embeddings from {emb_path}")
        else:
            print(f"[{dataset_name}] no precomputed embeddings found at {emb_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    graphs = [item["graph"] for item in batch]
    smiles = [item["smiles"] for item in batch]
    labels = [item["label"] for item in batch]
    smiles_embeddings = [item["smiles_embedding"] for item in batch]

    graph_batch = Batch.from_data_list(graphs)
    if labels[0].dim() == 0:
        label_batch = torch.stack(labels)
    else:
        label_batch = torch.stack(labels).to(torch.float32)

    has_precomputed = all(emb is not None for emb in smiles_embeddings)
    emb_batch = torch.stack(smiles_embeddings) if has_precomputed else None

    return {
        "graph": graph_batch,
        "smiles": smiles,
        "label": label_batch,
        "smiles_embedding": emb_batch,
    }


def get_dataset(
    dataset_name: str,
    data_dir: str = "dataset",
    embeddings_dir: str = "embeddings",
    task_type: Optional[str] = None,
) -> MolecularDataset:
    return MolecularDataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        embeddings_dir=embeddings_dir,
        task_type=task_type,
    )


def _murcko_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return smiles


def scaffold_split(
    dataset: MolecularDataset,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = 42,
) -> tuple:
    scaffold_to_indices: Dict[str, List[int]] = {}
    for idx, sample in enumerate(dataset.samples):
        key = _murcko_scaffold(sample["smiles"])
        scaffold_to_indices.setdefault(key, []).append(idx)

    n = len(dataset)
    train_cut = int(np.floor(frac_train * n))
    val_target = int(np.floor(frac_val * n))
    test_target = n - train_cut - val_target

    # Scaffolds bigger than half the val/test target are kept out of the
    # random shuffle (a single huge scaffold group would otherwise be able to
    # swamp a whole split); everything else is shuffled per-seed so that each
    # seed sees a different train/val/test assignment.
    index_sets = list(scaffold_to_indices.values())
    big_threshold = max(val_target, test_target) / 2
    big_sets = [s for s in index_sets if len(s) > big_threshold]
    small_sets = [s for s in index_sets if len(s) <= big_threshold]

    rng = np.random.RandomState(seed)
    rng.shuffle(big_sets)
    rng.shuffle(small_sets)
    groups = big_sets + small_sets

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for group in groups:
        if len(train_idx) < train_cut:
            train_idx.extend(group)
        elif len(val_idx) < val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, val_idx, test_idx


def random_split(
    dataset: MolecularDataset,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = 42,
) -> tuple:
    n = len(dataset)
    indices = np.arange(n)

    if dataset.task_type == "classification" and len(dataset.valid_labels) == n:
        labels = np.array(dataset.valid_labels)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - frac_train, random_state=seed)
        train_arr, rest_arr = next(sss.split(indices, labels))
        rest_labels = labels[rest_arr]
        val_frac = frac_val / (1.0 - frac_train)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_frac, random_state=seed + 1)
        rel_val, rel_test = next(sss2.split(range(len(rest_arr)), rest_labels))
        val_arr = rest_arr[rel_val]
        test_arr = rest_arr[rel_test]
        train_arr = indices[train_arr]
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        train_cut = int(frac_train * n)
        val_cut = int((frac_train + frac_val) * n)
        train_arr = perm[:train_cut]
        val_arr = perm[train_cut:val_cut]
        test_arr = perm[val_cut:]

    return list(map(int, train_arr)), list(map(int, val_arr)), list(map(int, test_arr))
