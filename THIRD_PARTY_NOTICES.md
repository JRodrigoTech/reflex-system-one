# Third-party notices

Reflex application source is distributed under the MIT License in [LICENSE](LICENSE). The Python runtimes, dependency distributions, model source code, and model weights retain their own terms. Model weights are downloaded from pinned upstream revisions and are not bundled in this repository.

The dependency inventory below covers every distinct distribution and version in `requirements/core.lock`, `requirements/dev.lock`, `requirements/laya.lock`, `requirements/decider.lock`, and `requirements/decision-cuda.lock` (51 distributions). Versions and permitted wheel hashes are pinned in those lock files. License identifiers and expressions come from the pinned distributions' package metadata; where an upstream only publishes a license classifier, the classifier is recorded in its SPDX-equivalent short form. NVIDIA CUDA components marked `LicenseRef-NVIDIA-Proprietary` are subject to the applicable NVIDIA CUDA Toolkit or component EULA.

## Pinned model repositories

| Upstream | Pinned revision | Declared license | Source |
|---|---|---|---|
| `convaiinnovations/laya` | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | Apache-2.0 | [model card and repository](https://huggingface.co/convaiinnovations/laya) |
| `Mapika/decider-2b` | `533964dae8be954c5b5e19fa4948e48408094c1e` | Apache-2.0 | [model card and repository](https://huggingface.co/Mapika/decider-2b) |
| `Mapika/decider-4b` | `eb5fbdfc9448473ec25e399882912863afbdb70e` | Apache-2.0 | [model card and repository](https://huggingface.co/Mapika/decider-4b) |
| `llm-semantic-router/Decision-1.0-Sol-2B` | `60ea30a48285ea097a9b3a728e71649b78331601` | Apache-2.0 | [model card and repository](https://huggingface.co/llm-semantic-router/Decision-1.0-Sol-2B) |
| `llm-semantic-router/Decision-1.0-Nox-4B` | `8901c750ccb8a2ba02f5ae026468ae22b4d12636` | Apache-2.0 | [model card and repository](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox-4B) |
| `llm-semantic-router/Decision-1.0-Lux-9B` | `bd45a30aee8c84032791c245c70f86dee5389cc8` | Apache-2.0 | [model card and repository](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) |

Model Manager retains upstream `README.md` and `LICENSE` files where provided. Installed model manifests record the repository, exact revision, declared license, and file checksums. Review the license file at the pinned revision before redistributing model files or a release bundle.

## Python distributions

| Distribution | Version | License | Upstream |
|---|---:|---|---|
| `annotated-doc` | 0.0.5 | MIT | [source](https://github.com/fastapi/annotated-doc) |
| `annotated-types` | 0.8.0 | MIT | [source](https://github.com/annotated-types/annotated-types) |
| `anyio` | 4.15.1 | MIT | [PyPI metadata](https://pypi.org/project/anyio/) |
| `certifi` | 2026.7.22 | MPL-2.0 | [source](https://github.com/certifi/python-certifi) |
| `click` | 8.5.0 | BSD-3-Clause | [source](https://github.com/pallets/click) |
| `colorama` | 0.4.6 | BSD | [source](https://github.com/tartley/colorama) |
| `einops` | 0.8.2 | MIT | [source](https://github.com/arogozhnikov/einops) |
| `fastapi` | 0.141.1 | MIT | [source](https://github.com/fastapi/fastapi) |
| `filelock` | 4.0.3 | MIT | [source](https://github.com/tox-dev/py-filelock) |
| `fla-core` | 0.5.2 | MIT | [source](https://github.com/fla-org/flash-linear-attention) |
| `flash-linear-attention` | 0.5.2 | MIT | [source](https://github.com/fla-org/flash-linear-attention) |
| `fsspec` | 2026.9.0 | BSD-3-Clause | [source](https://github.com/fsspec/filesystem_spec) |
| `h11` | 0.16.0 | MIT | [source](https://github.com/python-hyper/h11) |
| `hf-xet` | 1.6.0 | Apache-2.0 | [source](https://github.com/huggingface/xet-core) |
| `httpcore` | 1.0.9 | BSD-3-Clause | [source](https://github.com/encode/httpcore) |
| `httpx` | 0.28.1 | BSD-3-Clause | [source](https://github.com/encode/httpx) |
| `huggingface-hub` | 1.33.0 | Apache-2.0 | [source](https://github.com/huggingface/huggingface_hub) |
| `idna` | 3.20 | BSD-3-Clause | [source](https://github.com/kjd/idna) |
| `iniconfig` | 2.3.0 | MIT | [source](https://github.com/pytest-dev/iniconfig) |
| `jinja2` | 3.1.6 | BSD | [source](https://github.com/pallets/jinja) |
| `laya` | 0.3.20 | Apache-2.0 | [upstream repository](https://huggingface.co/convaiinnovations/laya) |
| `markdown-it-py` | 4.2.0 | MIT | [source](https://github.com/executablebooks/markdown-it-py) |
| `markupsafe` | 3.0.3 | BSD-3-Clause | [source](https://github.com/pallets/markupsafe) |
| `mdurl` | 0.1.2 | MIT | [source](https://github.com/executablebooks/mdurl) |
| `mpmath` | 1.3.0 | BSD | [source](https://github.com/mpmath/mpmath) |
| `networkx` | 3.7 | BSD-3-Clause | [source](https://github.com/networkx/networkx) |
| `numpy` | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | [license inventory](https://numpy.org/doc/stable/license.html) |
| `packaging` | 26.3 | Apache-2.0 OR BSD-2-Clause | [source](https://github.com/pypa/packaging) |
| `pluggy` | 1.6.0 | MIT | [source](https://github.com/pytest-dev/pluggy) |
| `pydantic` | 2.13.5 | MIT | [source](https://github.com/pydantic/pydantic) |
| `pydantic-core` | 2.46.5 | MIT | [source](https://github.com/pydantic/pydantic/tree/main/pydantic-core) |
| `pygments` | 2.21.0 | BSD-2-Clause | [source](https://github.com/pygments/pygments) |
| `pytest` | 9.1.1 | MIT | [source](https://github.com/pytest-dev/pytest) |
| `pytest-asyncio` | 1.4.0 | Apache-2.0 | [source](https://github.com/pytest-dev/pytest-asyncio) |
| `pyyaml` | 6.0.3 | MIT | [source](https://github.com/yaml/pyyaml) |
| `regex` | 2026.9.10 | Apache-2.0 AND CNRI-Python | [source](https://github.com/mrabarnett/mrab-regex) |
| `rich` | 15.0.0 | MIT | [source](https://github.com/Textualize/rich) |
| `safetensors` | 0.8.0 | Apache-2.0 | [source](https://github.com/huggingface/safetensors) |
| `setuptools` | 84.0.0 | MIT | [source](https://github.com/pypa/setuptools) |
| `shellingham` | 1.5.4 | ISC | [source](https://github.com/sarugaku/shellingham) |
| `starlette` | 1.7.0 | BSD-3-Clause | [source](https://github.com/Kludex/starlette) |
| `sympy` | 1.14.0 | BSD | [source](https://github.com/sympy/sympy) |
| `tokenizers` | 0.23.2 | Apache-2.0 | [source](https://github.com/huggingface/tokenizers) |
| `torch` | 2.14.0+cu130 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | [PyTorch license notices](https://github.com/pytorch/pytorch/tree/main/LICENSES) |
| `tqdm` | 4.70.1 | MPL-2.0 AND MIT | [source](https://github.com/tqdm/tqdm) |
| `transformers` | 5.17.0 | Apache-2.0 | [source](https://github.com/huggingface/transformers) |
| `triton-windows` | 3.8.0.post28 | MIT | [source](https://github.com/woct0rdho/triton-windows) |
| `typer` | 0.27.2 | MIT | [source](https://github.com/fastapi/typer) |
| `typing-extensions` | 4.16.0 | PSF-2.0 | [source](https://github.com/python/typing_extensions) |
| `typing-inspection` | 0.4.4 | MIT | [source](https://github.com/pydantic/typing-inspection) |
| `uvicorn` | 0.54.0 | BSD-3-Clause | [source](https://github.com/Kludex/uvicorn) |

## Portable Python runtimes

The Windows runtime uses a pinned CPython 3.12.14 `python-build-standalone` artifact from Astral. The distribution includes upstream notices and remains subject to the [python-build-standalone terms](https://github.com/astral-sh/python-build-standalone). Its artifact URL and SHA-256 digest are recorded in `manifests/runtime.json`.
