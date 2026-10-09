# Owner-local embeddings

NEXUS can generate semantic-memory vectors from an ONNX model that the owner has already placed on the machine. The runtime does not accept model URLs or remote model identifiers, does not download weights, and has no network client.

## Why this shape

The integration research reviewed `qdrant/fastembed` at upstream commit `539499b855478bcb6a810ac89a047de24c3b1866` (Apache-2.0) for its local ONNX embedding patterns. JARVIS does **not** vendor or copy FastEmbed source in this slice: its broader Hugging Face/download and multimodal dependency surface is unnecessary for the owner-local requirement. The implementation here is JARVIS-owned and keeps a smaller trust/runtime boundary.

## Install the optional runtime

```bash
python -m pip install -r requirements-local-embeddings.txt
```

The base JARVIS installation does not require these packages unless local embeddings are enabled.

## Model directory

The owner supplies a directory containing at minimum:

```text
my-embedding-model/
├── model.onnx
├── tokenizer.json
└── jarvis-embedding.json
```

`jarvis-embedding.json` is a JARVIS-owned compatibility manifest. Example:

```json
{
  "schema_version": 1,
  "model_id": "owner/my-embedding-model",
  "dimension": 384,
  "model_file": "model.onnx",
  "tokenizer_file": "tokenizer.json",
  "pooling": "mean",
  "normalize": true,
  "max_length": 512,
  "pad_token_id": 0,
  "output_name": "last_hidden_state"
}
```

`output_name` is optional. When omitted, JARVIS uses the first ONNX output. Supported pooling modes are `mean` and `cls`.

The manifest may only point to files inside the selected model directory. Dimension mismatches, unsupported ONNX input contracts, missing files, non-finite output, zero-length normalized vectors, or missing optional runtime packages fail closed.

## Usage

```python
from core.embedding_memory import TemporalEmbeddingMemory
from core.app_paths import user_data_dir
from core.local_embedding import LocalOnnxEmbeddingProvider
from core.temporal_memory import TemporalMemoryStore

store = TemporalMemoryStore(user_data_dir() / "memory" / "temporal.sqlite")
provider = LocalOnnxEmbeddingProvider(r"D:\AI\models\my-embedding-model")
memory = TemporalEmbeddingMemory(store, provider)

claim = store.add_claim(
    subject="Physics",
    predicate="topic",
    value="electric circuits",
    source="owner:notes",
)
memory.index_claim(claim.claim_id)
results = memory.semantic_search_text("circuits revision")
hybrid = memory.hybrid_search_text("physics circuits")
```

`TemporalEmbeddingMemory` initializes the existing in-process `VectorIndex` with the exact model dimension, supports bounded batch indexing, and can drive both semantic search and the hybrid BM25 + vector retriever.

## Authority and privacy

Embeddings are retrieval evidence only. They do not grant permission, dispatch actions, modify the Sovereign Owner Kernel, or bypass the Action Gateway. Model inference occurs against owner-supplied local files. The status contract explicitly reports `network_required: false`.

Durable temporal state requires an absolute path on a local fixed disk, outside the application source tree. JARVIS rejects redirected paths and protects the database, WAL, shared-memory, and vector-index files as owner-private state.
