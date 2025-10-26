# rag.py
import json
from qdrant_client import QdrantClient
from embedding import load_text_embedder
from qdrant_client.http.models import VectorParams
from typing import List

def ensure_collection(client: QdrantClient, collection: str, embedder) -> None:
    """Ensure the Qdrant collection exists, otherwise create it."""
    existing = [col.name for col in client.get_collections().collections]
    if collection not in existing:
        print(f"Collection '{collection}' not found. Creating it...")

        # Determine embedding vector size
        test_vec = embedder.encode(["test"])[0]
        vector_size = test_vec.shape[0] if hasattr(test_vec, "shape") else len(test_vec)

        # New Qdrant client requires vectors_config
        vectors_config = VectorParams(size=vector_size, distance="Cosine")

        client.recreate_collection(
            collection_name=collection,
            vectors_config=vectors_config
        )
        print(f"Collection '{collection}' created.")


def retrieve_docs(question: str, table_metadata: dict, client: QdrantClient, embedder, collection: str = "questdb_docs", top_k: int = 3) -> List[str]:
    """Encode the question and retrieve top-k chunks from Qdrant."""
    # Ensure collection exists
    ensure_collection(client, collection, embedder)

    # Encode the question
    vecs = embedder.encode([question])
    query_vec = vecs[0].tolist() if hasattr(vecs, "tolist") else list(vecs[0])

    # Search in Qdrant
    results = client.search(collection_name=collection, query_vector=query_vec, limit=top_k)

    # Extract chunk_text from payload
    docs = [
        getattr(r.payload, "get", lambda k, d=None: r.payload[k])("chunk_text")
        if isinstance(r.payload, dict) else r.payload["chunk_text"]
        for r in results
    ]
    return docs


def generate_answer(question: str, table_metadata: dict, docs: list, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Generate answer using a causal LLM (GPU preferred)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import os
    import requests

    prompt = f"""
You are an expert on QuestDB. Answer the question using ONLY the provided documentation.

Documentation:
{chr(10).join(docs)}

Table Schema:
{json.dumps(table_metadata)}

Question: {question}
Answer:
"""

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        hf_token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
        if hf_token:
            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "return_full_text": False}}
            url = f"https://api-inference.huggingface.co/models/{model_name}"
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                raise RuntimeError(f"HF Inference returned unexpected response: {data}")
            except Exception as e:
                raise RuntimeError(f"Failed to call Hugging Face Inference API: {e}")

    # Local model loading
    def _looks_like_large_model(name: str) -> bool:
        nl = name.lower()
        return "qwen" in nl or "1.7b" in nl or "0.6b" in nl

    # Allow user to force attempting the requested model on CPU. If not set, we'll prefer a safe CPU-friendly fallback
    force_cpu = os.getenv("FORCE_LLM_ON_CPU", "").lower() in ("1", "true", "yes")

    if torch.cuda.is_available():
        model_to_load = model_name
        torch_dtype = torch.float16
        offload_folder = os.path.join(os.getcwd(), "hf_offload")
        os.makedirs(offload_folder, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load, trust_remote_code=True, device_map="auto", torch_dtype=torch_dtype, offload_folder=offload_folder
        )
    else:
        # No CUDA. Choose a safe CPU model unless the user explicitly forced the requested model on CPU.
        if not force_cpu and _looks_like_large_model(model_name):
            fallback_cpu_model = os.getenv("CPU_FRIENDLY_LLM", "distilgpt2")
            print(
                f"No CUDA device detected and no HF token; requested model '{model_name}' looks large. "
                f"Falling back to CPU-friendly model '{fallback_cpu_model}'."
                " Set FORCE_LLM_ON_CPU=1 to attempt loading the requested model on CPU (may fail/out-of-memory)."
            )
            model_to_load = fallback_cpu_model
        else:
            model_to_load = model_name

        torch_dtype = torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load, trust_remote_code=True, device_map={"": "cpu"}, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to model device
    try:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception:
        pass

    # Generation parameters: sampling + penalties to discourage looping/repetition
    gen_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "num_return_sequences": 1,
    }
    # Ensure eos/pad token ids are set to avoid warnings and to let generation stop
    try:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    except Exception:
        pass
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(**inputs, **gen_kwargs)

    # The generated sequence contains the prompt followed by new tokens. Slice to get only the completion.
    import torch as _torch
    out_ids = outputs[0]
    if hasattr(inputs, "get") and inputs.get("input_ids") is not None:
        input_len = inputs["input_ids"].shape[1]
        # If output length is greater than input length, slice the new tokens; otherwise decode whole output.
        if out_ids.shape[0] == 0:
            answer = ""
        else:
            if out_ids.shape[0] > input_len:
                gen_ids = out_ids[input_len:]
            else:
                gen_ids = out_ids
            # Ensure tensor on CPU for tokenizer
            if isinstance(gen_ids, _torch.Tensor):
                gen_ids = gen_ids.cpu()
            answer = tokenizer.decode(gen_ids, skip_special_tokens=True)
    else:
        # Fallback: decode entire output
        answer = tokenizer.decode(out_ids, skip_special_tokens=True)

    # Post-process to collapse pathological repetition like "The answer is simple. The answer is simple..."
    def _collapse_repeats(text: str) -> str:
        # Collapse consecutive identical sentences or lines
        import re

        # Split on sentence-ending punctuation while keeping it
        parts = re.split(r'([.!?]\s+)', text)
        if len(parts) <= 1:
            # fallback: collapse repeated lines
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            out_lines = []
            prev = None
            for l in lines:
                if l == prev:
                    continue
                out_lines.append(l)
                prev = l
            return "\n".join(out_lines)

        # Reconstruct sentences and collapse identical consecutive sentences
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sent = (parts[i] + parts[i + 1]).strip()
            if sent:
                sentences.append(sent)
        # If there's a trailing fragment
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        out = []
        prev = None
        for s in sentences:
            if s == prev:
                # skip repeated sentence
                continue
            out.append(s)
            prev = s
        return " ".join(out)

    answer = _collapse_repeats(answer)
    return answer
