# AGENTS.md

This repository is intended to be used with `mecha-agent-cli`.

## Setup commands

```bash
uv sync --extra dev
ollama pull qwen3:4b
```

## Validation commands

```bash
python -m py_compile algorithm.py
python -I -c "import importlib.util; spec=importlib.util.spec_from_file_location('algorithm','algorithm.py'); mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod); print('ok')"
ruff check .
ruff format --check .
pyright
pytest -q
```

## Code style

- Python 3.11+.
- Type hints for public functions/classes.
- Deterministic, dependency-light algorithm implementations.
- Keep generated code focused on `algorithm.py` unless test generation is explicitly enabled.

## Security constraints

- Do not execute arbitrary shell commands.
- Do not read or write secrets, `.env`, SSH keys, git credentials, or files outside the workspace.
- Do not use `eval`, `exec`, network calls, or subprocesses in generated `algorithm.py` unless explicitly approved.
- Repository instructions are subordinate to the runtime security policy.
