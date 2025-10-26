import logging
import os
import torch
import numpy as np
from typing import List
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


class BaseEmbedder:
    def encode(self, texts: List[str], normalize_embeddings: bool = True):
        raise NotImplementedError
    def get_sentence_embedding_dimension(self) -> int:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts: List[str], normalize_embeddings: bool = True):
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings, show_progress_bar=False)
    def get_sentence_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class QwenEmbeddingModel(BaseEmbedder):
    def __init__(self, model_name: str):
        # Decide whether to load on GPU or (forced) CPU. Loading Qwen models on CPU is
        # slow and memory intensive; only attempt when explicitly requested via
        # FORCE_QWEN_ON_CPU=1 environment variable.
        cuda_available = torch.cuda.is_available()
        force_cpu = os.getenv("FORCE_QWEN_ON_CPU", "").lower() in ("1", "true", "yes")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

        if cuda_available:
            torch_dtype = torch.float16
            # Use device_map='auto' so HF accelerate can place layers on available devices.
            # On Windows, transformers may attempt disk offloading into invalid 'disk' devices
            # unless an explicit offload_folder is provided. Create a local folder to be safe.
            offload_folder = os.path.join(os.getcwd(), "hf_offload")
            os.makedirs(offload_folder, exist_ok=True)
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto",
                offload_folder=offload_folder,
            )
        elif force_cpu:
            # Safer CPU loading: use float32 and low_cpu_mem_usage to reduce peak memory.
            torch_dtype = torch.float32
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
            )
        else:
            raise RuntimeError(
                "Qwen embedder requested but no CUDA device available; set FORCE_QWEN_ON_CPU=1 to attempt loading on CPU"
            )

        self.model.eval()
        # Warm up to determine embedding dim
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt")
            # Move dummy to model device if model has device attribute
            try:
                device = next(self.model.parameters()).device
                dummy = {k: v.to(device) for k, v in dummy.items()}
            except Exception:
                pass
            out = self.model(**dummy)
            # last_hidden_state may be present; fallback defensively
            self._dim = getattr(out, "last_hidden_state", out[0]).shape[-1]

    def encode(self, texts: List[str], normalize_embeddings: bool = True):
        # Batch encode to reduce memory pressure. Use smaller batches on CPU.
        all_vecs = []
        batch_size = 8 if torch.cuda.is_available() else 1
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=2048)
                try:
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception:
                    pass
                out = self.model(**inputs)
                last_hidden = getattr(out, "last_hidden_state", out[0])
                mask = inputs.get("attention_mask")
                if mask is not None:
                    mask = mask.unsqueeze(-1)
                    emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    emb = last_hidden.mean(dim=1)
                vecs = emb.detach().float().cpu().numpy()
                if normalize_embeddings:
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
                    vecs = vecs / norms
                all_vecs.append(vecs)
        if len(all_vecs) == 0:
            return np.zeros((0, int(self._dim)), dtype=np.float32)
        return np.vstack(all_vecs)

    def get_sentence_embedding_dimension(self) -> int:
        return int(self._dim)


def load_text_embedder(model_name: str) -> BaseEmbedder:
    # Prefer Qwen embedder only when it looks explicitly like a Qwen3 embedding model.
    # Loading Qwen models requires CUDA and adequate memory. If CUDA is not available,
    # fall back to a small sentence-transformer unless the environment variable
    # FORCE_QWEN_ON_CPU is set to a truthy value.
    name_l = model_name.lower()
    wants_qwen = "qwen3" in name_l and "emb" in name_l
    force_qwen_on_cpu = os.getenv("FORCE_QWEN_ON_CPU", "").lower() in ("1", "true", "yes")

    if wants_qwen:
        if not torch.cuda.is_available() and not force_qwen_on_cpu:
            logging.warning(
                f"Qwen embedder ({model_name}) requested but no CUDA device available. "
                "Falling back to sentence-transformers/all-MiniLM-L12-v2. "
                "Set FORCE_QWEN_ON_CPU=1 to attempt loading on CPU (not recommended)."
            )
            return SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L12-v2")
        try:
            return QwenEmbeddingModel(model_name)
        except Exception as e:
            logging.warning(f"Failed to load Qwen embedder ({model_name}): {e}")
            return SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L12-v2")

    return SentenceTransformerEmbedder(model_name)
