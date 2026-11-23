import abc
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

import esm
import numpy as np
import torch
import torch.nn.functional as F
from Bio.Align import substitution_matrices
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModel, AutoTokenizer, EsmTokenizer

from plm_developability.utils_basic import mkdir_if_missing
from models.SaProt.model.saprot.base import SaprotBaseModel
from models.SaProt.utils.foldseek_util import get_struc_seq


# fill out-of-alphabet amminoacid
def replace_ammino(x, alphabet, most_common_aa):  # noqa: D103
    residues = list(set(x))
    out_of_alphabet = [a for a in residues if a not in alphabet]
    for ss in out_of_alphabet:
        x = x.replace(ss, most_common_aa)
    return x


class Encoding(abc.ABC):
    """Abstract base class for encodings."""

    def __init__(self, dd_dict):
        """Initialize encoding.

        Args:
            dd_dict: device and dtype in dict
        """
        self.device = dd_dict["device"]
        self.dtype = dd_dict["dtype"]

    @abc.abstractmethod
    def encode_sequences(self, sequences: np.ndarray) -> torch.Tensor:
        """Encode sequences.

        Args:
            sequences: numpy array of sequences

        Returns
        -------
            one-hot encoded sequences as torch Tensor
        """
        raise NotImplementedError

    def score_sequences(self, sequences, wt_seq: str) -> torch.Tensor:
        """Compute WT-relative masked pseudo-log-likelihood LLR for each sequence.

        Args:
            sequences: list of VH-VL sequences (heavy-light format with '-' separator)
            wt_seq: wildtype reference sequence in VH-VL format

        Returns
        -------
            1-D float32 tensor of LLR scores, one per input sequence
        """
        raise NotImplementedError(f"score_sequences not implemented for {type(self).__name__}")


class ESMEncoding(Encoding):
    """ESMEncoding subclass implementing the abstract encode_sequences method."""

    def __init__(self, dd_dict, model_version, chunk_size=100):
        """Initialize ESM encoding with device and dtype information."""
        super().__init__(dd_dict)  # Call the constructor of the base class
        self.model, self.alphabet = self.load_esm_model(model_version, self.device)
        self.chunk_size = chunk_size
        self.model_version = model_version

    def __str__(self) -> str:
        """String identifier of encoding."""
        tags = self.model_version.split("_")[1:-1]
        return f"ESM-{'-'.join(tags)}"

    def load_esm_model(self, model_version, device):
        """Load the specified ESM model version.

        Args:
            model_version: ESM model version string

        Returns
        -------
            The loaded ESM model and its associated alphabet
        """
        # Dynamically call the function to load the model based on the version string
        model_loader = getattr(esm.pretrained, model_version)
        model, alphabet = model_loader()
        # Move model to specified device
        model = model.to(torch.device(device))
        return model, alphabet

    def encode_sequences(self, sequences, repr_layers=6) -> torch.Tensor:
        """Encode sequences using the ESM model with handling for batch size.

        Args:
            sequences: List of sequences
            repr_layers: Representation layer of the model to use
            batch_size: Number of sequences to process per batch

        Returns
        -------
            Torch tensor of concatenated encoded sequences
        """
        batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # Disables dropout for deterministic results

        embeddings_list = []

        n_batches = (len(sequences) + self.chunk_size - 1) // self.chunk_size
        for start_idx in tqdm(range(0, len(sequences), self.chunk_size),
                              desc="ESM encoding", unit="batch", total=n_batches):
            # Create a mini-batch
            batch_sequences = sequences[start_idx : start_idx + self.chunk_size]
            # Split heavy and light chains
            split_sequences = [
                (str(idx), [part for part in seq.split("-") if part]) for idx, seq in enumerate(batch_sequences)
            ]
            # Reconstruct sequences — sequence selection/cutting is handled upstream (benchmark script)
            batch_data = [(str(idx), "".join(seq_parts)) for idx, seq_parts in split_sequences]

            _, _, batch_tokens = batch_converter(batch_data)

            with torch.no_grad():
                results = self.model(batch_tokens.to(self.device), repr_layers=[repr_layers], return_contacts=False)
            sequence_embeddings = results["representations"][repr_layers]

            # Collect and reduce embeddings
            reduced_embeddings = sequence_embeddings[:, 1:-1].mean(1)

            embeddings_list.append(reduced_embeddings)

        # Concatenate all batch results into a single tensor
        return torch.cat(embeddings_list, dim=0)

    def score_sequences(self, sequences, wt_seq: str) -> torch.Tensor:
        """Compute WT-relative masked pseudo-log-likelihood LLR for each sequence.

        WT masked log-probs are precomputed once (batched by chunk_size), then each
        variant's mutated positions are scored against the same WT reference.
        """
        batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

        wt_clean = "".join(p for p in wt_seq.split("-") if p)
        wt_len = len(wt_clean)

        # Stage 1 — WT reference log-probs.
        # For each position i, mask position i in the WT sequence and run one forward pass.
        # Store log P(WT_i | WT_{-i}). Positions are batched in groups of chunk_size.
        # logits shape from ESM: [batch, seq_len+2, vocab]; +2 for <cls>/<eos>, so pos i → token i+1.
        logging.info(f"[ESM score_sequences] Precomputing WT log-probs: {wt_len} positions")
        wt_log_probs: dict = {}
        for chunk_start in range(0, wt_len, self.chunk_size):
            positions = list(range(chunk_start, min(chunk_start + self.chunk_size, wt_len)))
            batch_data = [
                (str(p), wt_clean[:p] + "<mask>" + wt_clean[p + 1:]) for p in positions
            ]
            _, _, tokens = batch_converter(batch_data)
            with torch.no_grad():
                results = self.model(tokens.to(self.device), repr_layers=[], return_contacts=False)
            for i, pos in enumerate(positions):
                log_p = F.log_softmax(results["logits"][i, pos + 1], dim=-1)
                wt_log_probs[pos] = log_p[self.alphabet.get_idx(wt_clean[pos])].item()

        # Stage 2 — Variant scoring.
        # For each variant, find positions that differ from WT (mutated positions only).
        # For each mutated position, mask that position in the variant context (not WT),
        # so the model conditions on the variant's other residues.
        # LLR += log P(var_i | var_{-i}) - wt_log_probs[i]
        logging.info(f"[ESM score_sequences] Scoring {len(sequences)} variants")
        seq_cleans = ["".join(p for p in seq.split("-") if p) for seq in sequences]
        scores = []
        for seq_clean in seq_cleans:
            mutated = [
                p for p in range(min(len(seq_clean), wt_len)) if seq_clean[p] != wt_clean[p]
            ]
            llr = 0.0
            for pos in mutated:
                masked = seq_clean[:pos] + "<mask>" + seq_clean[pos + 1:]
                _, _, tokens = batch_converter([(0, masked)])
                with torch.no_grad():
                    results = self.model(tokens.to(self.device), repr_layers=[], return_contacts=False)
                log_p = F.log_softmax(results["logits"][0, pos + 1], dim=-1)
                llr += log_p[self.alphabet.get_idx(seq_clean[pos])].item() - wt_log_probs[pos]
            scores.append(llr)

        return torch.tensor(scores, dtype=torch.float32)


def contact_w_embedding(sequence_embeddings, contacts):
    """Combine contact and sequence embeddings.

    Args:
        sequence_embeddings: Tensor of sequence embeddings
        contacts: Tensor of contact information
        # sequence_embeddings: [batch, seq_len, emb_dim]
        # contacts: [batch, seq_len, seq_len]
    """
    # Example: For each residue, sum its contacts to get a "contact score"
    # Compute weighted sum of embeddings using contact matrix as weights
    # contacts: [batch, seq_len, seq_len], sequence_embeddings: [batch, seq_len, emb_dim]
    # If contacts is already an attention matrix (values between 0 and 1, rows sum to 1), normalization is not needed.
    # contact_weights = contacts / (contacts.sum(dim=2, keepdim=True) + 1e-8)  # [batch, seq_len, seq_len]
    # Weighted sum: for each residue, sum over all other residues' embeddings weighted by contact_weights
    weighted_embedding = torch.matmul(contacts, sequence_embeddings)  # [batch, seq_len, emb_dim]
    # Reduce over sequence length (mean)
    weighted_embedding = weighted_embedding.mean(dim=1)  # [batch, emb_dim]
    return weighted_embedding


class SaprotEncoding(Encoding):
    """A class to handle Saprot (https://github.com/westlake-repl/SaProt)
    embeddings for antibody sequences.
    """  # noqa: D205

    def __init__(
        self,
        dd_dict,
        use_structure,
        model_dir,
        saprot_version,
        chain_mapper: Optional[Dict] = None,
        chunk_size=100,
    ):
        """Initialize Saprot encoding with device and dtype information."""
        super().__init__(dd_dict)

        # Load models and move them to the specified device
        config = {
            "task": "base",
            "config_path": model_dir,  # Note this is the directory path of SaProt, not the ".pt" file
            "load_pretrained": True,
        }

        self.model = SaprotBaseModel(**config)
        self.model = self.model.to(torch.device(self.device))
        self.tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
        self.chunk_size = chunk_size
        self.chain_mapper = chain_mapper
        self.use_structure = use_structure
        self.saprot_version = saprot_version
        self._model_dir = model_dir

    def __str__(self) -> str:
        """String identifier of encoding."""
        if self.use_structure in ["full", "struct"]:
            return f"Saprot-struct-{self.saprot_version}"
        elif self.use_structure == "seq":
            return f"Saprot-seq-{self.saprot_version}"
        else:
            raise ValueError("Invalid Saprot type. Choose from 'full' or 'seq'")

    def struc_to_struc_sec(self, pdb_files: List[str], data_dir, split) -> List[str]:
        """Generate struc_seq from pdb files.

        Args:
            pdb_files: List of pdb_files

        Returns
        -------
            List of struc_seq to be used for saprot model
        """
        heavy_path = os.path.join(data_dir, f"struc_seqs/struc_seq_heavy_{split}.json")
        light_path = os.path.join(data_dir, f"struc_seqs/struc_seq_light_{split}.json")
        mkdir_if_missing(heavy_path)
        mkdir_if_missing(light_path)

        # If both files exist, load and return
        if os.path.exists(heavy_path) and os.path.exists(light_path):
            with open(heavy_path, "r") as f:
                struc_seq_list_heavy = json.load(f)
            with open(light_path, "r") as f:
                struc_seq_list_light = json.load(f)
            return struc_seq_list_heavy, struc_seq_list_light

        # Otherwise, generate and save
        struc_seq_list_heavy = []
        struc_seq_list_light = []
        # Compute absolute path to foldseek binary relative to package location
        foldseek_path = Path(__file__).resolve().parents[1] / "models" / "SaProt" / "bin" / "foldseek"
        for pdb_file in pdb_files:
            if not os.path.exists(pdb_file):
                raise FileNotFoundError(f"PDB file {pdb_file} does not exist.")
            # # Extract the "A" chain from the pdb file and encode it into a struc_seq
            # pLDDT is used to mask low-confidence regions if "plddt_mask" is True. Please set it to True when
            # use AF2 structures for best performance.
            parsed_seqs = get_struc_seq(
                str(foldseek_path), pdb_file, [self.chain_mapper["H"], self.chain_mapper["L"]], plddt_mask="auto"
            )
            _, _, struc_seq_heavy = parsed_seqs[self.chain_mapper["H"]]
            _, _, struc_seq_light = parsed_seqs[self.chain_mapper["L"]]

            struc_seq_list_heavy.append(struc_seq_heavy)
            struc_seq_list_light.append(struc_seq_light)

        # Save results
        with open(heavy_path, "w") as f:
            json.dump(struc_seq_list_heavy, f)
        with open(light_path, "w") as f:
            json.dump(struc_seq_list_light, f)

        return struc_seq_list_heavy, struc_seq_list_light

    def seq_to_struc_sec(self, sequences: list) -> tuple:
        """
        For each sequence in the input list, split into heavy and light chains,
        add '#' after each character, and return two lists.

        Args:
            sequences: List of strings, each like 'AMKH-WVKH'

        Returns
        -------
            struc_seq_list_heavy: List of processed heavy chain sequences
            struc_seq_list_light: List of processed light chain sequences
        """  # noqa: D205
        struc_seq_list_heavy = []
        struc_seq_list_light = []
        for seq in sequences:
            parts = seq.split("-")
            heavy = parts[0] if len(parts) > 0 else ""
            light = parts[1] if len(parts) > 1 else ""
            heavy_seq = "#".join(list(heavy)) + "#" if heavy else ""
            light_seq = "#".join(list(light)) + "#" if light else ""
            struc_seq_list_heavy.append(heavy_seq)
            struc_seq_list_light.append(light_seq)
        return struc_seq_list_heavy, struc_seq_list_light

    def encode_sequences(self, sequences, split=None, pdb_files=None, data_dir=None):
        """Encode sequences in batches using the Saprot models.

        Args:
            sequences: List of antibody sequences, with VL and VH concatenated in one string
            pdb_files: List of pdb file locations
            data_dir: Directory where the struc_seqs are stored
            split: text train/test for saving the struc sec

        Returns
        -------
            Torch tensor with the embedding results
        """
        if pdb_files and self.use_structure == "full":
            sequences_heavy, sequences_light = self.struc_to_struc_sec(pdb_files, data_dir, split)
        else:
            sequences_heavy, sequences_light = self.seq_to_struc_sec(sequences)

        sequences_vh_vl = [heavy_seq + light_seq for heavy_seq, light_seq in zip(sequences_heavy, sequences_light)]

        self.model.eval()  # Disables dropout for deterministic results

        embeddings_list = []

        n_batches = (len(sequences_vh_vl) + self.chunk_size - 1) // self.chunk_size
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(sequences_vh_vl), self.chunk_size),
                                  desc="SaProt encoding", unit="batch", total=n_batches):
                # Create a mini-batch
                batch_sequences = sequences_vh_vl[start_idx : start_idx + self.chunk_size]
                seq_tokenized = self.tokenizer(batch_sequences, return_tensors="pt", padding=True)
                seq_tokenized = {k: v.to(self.device) for k, v in seq_tokenized.items()}

                seqcodings_h_l = self.model.get_hidden_states(seq_tokenized, reduction="mean")
                seqcodings_h_l = torch.stack(seqcodings_h_l, dim=0).cpu()
                embeddings_list.append(seqcodings_h_l)
                # Free up GPU memory
                del seqcodings_h_l, seq_tokenized
                torch.cuda.empty_cache()

        # Concatenate all batch results into a single tensor
        return torch.cat(embeddings_list, dim=0)

    def score_sequences(self, sequences, wt_seq: str) -> torch.Tensor:
        """Compute WT-relative masked pseudo-log-likelihood LLR using SaProt seq-only mode.

        Lazy-loads a fresh EsmForMaskedLM (with LM head intact) on the first call, since
        SaprotBaseModel removes its lm_head for embedding extraction.
        """
        from transformers import EsmForMaskedLM

        if not hasattr(self, "_lm_model"):
            self._lm_model = EsmForMaskedLM.from_pretrained(self._model_dir).to(self.device)
            self._lm_model.eval()

        # Convert WT to Saprot seq-only format and get combined AA string
        wt_heavy_s, wt_light_s = self.seq_to_struc_sec([wt_seq])
        wt_saprot = wt_heavy_s[0] + wt_light_s[0]
        wt_aa = "".join(p for p in wt_seq.split("-") if p)

        mask_id = self.tokenizer.mask_token_id
        wt_enc = self.tokenizer(wt_saprot, return_tensors="pt")
        wt_input_ids = wt_enc["input_ids"]  # [1, L+2]: +2 for [CLS]/[EOS]

        # Stage 1 — WT reference log-probs.
        # Each Saprot token is an "AA#" pair (e.g. "E#"); position i in the AA string maps to
        # token index i+1 (offset by [CLS]). Mask that token, run forward, store log P(WT_i | WT_{-i}).
        logging.info(f"[Saprot score_sequences] Precomputing WT log-probs: {len(wt_aa)} positions")
        wt_log_probs: dict = {}
        for pos, aa in enumerate(wt_aa):
            tok_pos = pos + 1  # [CLS] sits at index 0
            masked_ids = wt_input_ids.clone()
            masked_ids[0, tok_pos] = mask_id
            with torch.no_grad():
                logits = self._lm_model(input_ids=masked_ids.to(self.device)).logits
            log_p = F.log_softmax(logits[0, tok_pos], dim=-1)
            wt_token_id = self.tokenizer.convert_tokens_to_ids(f"{aa}#")
            wt_log_probs[pos] = log_p[wt_token_id].item()

        # Stage 2 — Variant scoring.
        # Find mutated positions, mask each in the variant's own token sequence (not WT),
        # so the model conditions on the variant's context.
        # LLR += log P(var_i# | var_{-i}) - wt_log_probs[i]
        logging.info(f"[Saprot score_sequences] Scoring {len(sequences)} variants")
        scores = []
        for seq in sequences:
            var_heavy_s, var_light_s = self.seq_to_struc_sec([seq])
            var_saprot = var_heavy_s[0] + var_light_s[0]
            var_aa = "".join(p for p in seq.split("-") if p)
            var_enc = self.tokenizer(var_saprot, return_tensors="pt")
            var_input_ids = var_enc["input_ids"]

            mutated = [i for i in range(min(len(var_aa), len(wt_aa))) if var_aa[i] != wt_aa[i]]
            llr = 0.0
            for pos in mutated:
                tok_pos = pos + 1
                masked_ids = var_input_ids.clone()
                masked_ids[0, tok_pos] = mask_id
                with torch.no_grad():
                    logits = self._lm_model(input_ids=masked_ids.to(self.device)).logits
                log_p = F.log_softmax(logits[0, tok_pos], dim=-1)
                var_token_id = self.tokenizer.convert_tokens_to_ids(f"{var_aa[pos]}#")
                llr += log_p[var_token_id].item() - wt_log_probs[pos]
            scores.append(llr)

        return torch.tensor(scores, dtype=torch.float32)

class OneHotEncoding(Encoding):
    """ Get one-hot encoding of seqences"""

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, dtype=int)
        self.char_to_index = {}
        self.index_to_char = {}

    def __str__(self) -> str:
        """String identifier of encoding."""  # noqa: D401
        return "OneHotEncoding"

    def _fit_and_transform_sequences(self, sequences):
        """
        Fit the encoder on the sequences and compute padded one-hot encoding.

        Args:
            sequences: List of sequences where each sequence is a string
                       containing heavy and light chains concatenated with "-".

        Returns
        -------
            Padded one-hot encoded NumPy arrays for each sequence.
        """
        # Flatten the list of sequences to extract individual characters, ignoring "-"
        flat_chars = [char for seq in sequences for char in seq if char != '-']

        # Fit the encoder on the unique characters found in actual sequences
        self.encoder.fit([[char] for char in set(flat_chars)])

        # Store the mapping of characters to indices
        self.char_to_index = {char: idx for idx,
                              char in enumerate(self.encoder.categories_[0])}
        self.index_to_char = {idx: char for char,
                              idx in self.char_to_index.items()}

        # Encode sequences as a whole, treating them as unified strings
        encoded_seqs = [
            self.encoder.transform([[char] for char in seq if char != '-'])
            for seq in sequences
        ]

        # Determine the length of the longest encoded sequence
        max_length = max(len(encoded_seq) for encoded_seq in encoded_seqs)

        # Pad the encoded sequences to have uniform length
        padded_encodings = [
            np.pad(encoded_seq, ((0, max_length - len(encoded_seq)),
                   (0, 0)), 'constant', constant_values=0)
            for encoded_seq in encoded_seqs
        ]

        return np.array(padded_encodings)

    def encode_sequences(self, sequences, return_mean=True):
        """
        Process the padded encodings to get either the mean or full vectors.

        Args:
            sequences: List of sequences where each sequence is a string
                       containing heavy and light chains concatenated with "-".
            return_mean: If True, return mean encoding; otherwise, return full padded tensor.

        Returns:
            A PyTorch tensor with the specified encodings.
        """
        padded_encodings = self._fit_and_transform_sequences(sequences)
        if return_mean:
            # Calculate the mean one-hot encoding for each sequence
            mean_encodings = np.mean(padded_encodings, axis=1)

            # Apply threshold to drop columns with more than 99% zeros
            threshold = 0.99
            cols_to_drop = [
                col_index for col_index in range(mean_encodings.shape[1])
                if (mean_encodings[:, col_index] == 0).sum() / mean_encodings.shape[0] > threshold
            ]
            reduced_encodings = np.delete(mean_encodings, cols_to_drop, axis=1)

            # Convert the reduced encodings to a PyTorch tensor
            return torch.tensor(reduced_encodings, dtype=torch.float32)
        else:
            # Flatten each sequence's encoding to ensure coherent size
            n_sequences, max_length, encoding_dim = padded_encodings.shape
            full_encodings = padded_encodings.reshape((n_sequences, -1))

            return torch.tensor(full_encodings, dtype=torch.float32)
