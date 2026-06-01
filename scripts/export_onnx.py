"""
waggle-mcp export onnx
============================
Loads the all-MiniLM-L6-v2 model via SentenceTransformer and exports it to ONNX format (either via backend="onnx" or optimum).

Usage
-----

Requirements: pip install --upgrade onnx onnxscript sentence-transformers optimum[onnx]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime
from waggle.embeddings import EmbeddingModel
import numpy as np
import time
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Load the model
# ---------------------------------------------------------------------------
embed_model = EmbeddingModel("deterministic")

# ---------------------------------------------------------------------------
# Export as ONNX + saving ONNX
# ---------------------------------------------------------------------------
onnx_program = torch.onnx.export(model=embed_model, dynamo=True)
onnx_program.save("embeddings_model.onnx")

# ---------------------------------------------------------------------------
# Comparison - original model vs. ONNX-loaded model
# ---------------------------------------------------------------------------

contents = [
    "We decided to use PostgreSQL as the primary database for the Acme web app. PostgreSQL offers ACID compliance, rich JSON support, and scales well for our expected load.",
    "PostgreSQL was chosen because the team has prior experience with it, it supports JSONB for flexible schema evolution, and the managed RDS offering fits our AWS deployment plan.",
    "For local development, we switched to SQLite to eliminate the Docker dependency and speed up onboarding. Production still targets PostgreSQL.",
    "New engineers were spending 30+ minutes setting up Postgres locally. SQLite requires zero setup and the ORM abstracts the difference.",
    "Final decision: use PostgreSQL in all environments (dev, staging, prod) via Docker Compose. The SQLite shortcut caused subtle migration drift. We added a one-command `make dev-up` to remove the setup friction.",
    "SQLite and PostgreSQL handle NULL semantics, JSON operators, and transaction isolation differently. Two bugs in staging traced back to SQLite-only dev. Docker Compose with a health-check solves onboarding without sacrificing parity.",
    "We will use Auth0 for authentication and SSO. Rolling our own OAuth is out of scope for v1. Auth0 supports SAML for enterprise customers.",
    "The team has no dedicated security engineer. Auth0 handles MFA, breach detection, and compliance certifications. Estimated 3-week saving vs. building in-house.",
    "We will deploy the web app on AWS ECS using Fargate. No EC2 instance management, auto-scaling, and integrates with our existing AWS account.",
    "The team is 4 engineers. Fargate means no patching, no AMI management, and cost scales to zero when idle. Kubernetes was considered but deemed over-engineered for v1.",
    "The team prefers TypeScript over plain JavaScript for all frontend work. Strict mode enabled. No `any` without a comment explaining why.",
    "The team prefers dark mode as the default UI theme. Light mode should be available as a toggle but dark is the out-of-box experience.",
    "The team strongly prefers small, focused PRs — ideally under 400 lines. Large PRs block review and increase merge conflict risk. Feature flags are the preferred mechanism for shipping incomplete features.",
    "The Acme web app team has 4 engineers: 2 full-stack, 1 backend, 1 frontend/design. No dedicated DevOps or security engineer.",
    "The target public launch date is end of Q3. The v1 scope is intentionally narrow: auth, core CRUD, and basic reporting. v2 will add integrations and advanced analytics."
]

# Option 1: original embedding model created from the EmbeddingModel class, embeddings extracted one by one
embeddings_original = np.array([])
start_time = time.perf_counter()
for c in contents:
    e = embed_model.embed(c)
    embeddings_original = np.append(embeddings_original, e)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Original model: embeddings obtained in {elapsed_time:.6f} seconds")

# Option 2: original embedding model created from the EmbeddingModel class, batch-processed embeddings extraction
start_time = time.perf_counter()
embeddings_batched = embed_model.embed_batch(contents)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Original model with batching: embeddings obtained in {elapsed_time:.6f} seconds")

# Loading the saved ONNX file
onnx_model = onnx.load("embeddings_model.onnx")
onnx.checker.check_model(onnx_model)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model_inputs = [tokenizer(c, return_tensors="np") for c in contents]
ort_session = onnxruntime.InferenceSession(
    "embeddings_model.onnx", providers=["CPUExecutionProvider"]
)
onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), model_inputs)}

# Option 3: loaded ONNX model, embeddings extracted one by one
start_time = time.perf_counter()
output_onnx = onnx_model.run(None, dict(model_inputs))
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Original model: embeddings obtained in {elapsed_time:.6f} seconds")



