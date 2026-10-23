# Security policy

Please do not report security problems in public issues.

Use **Report a vulnerability** on the Security tab of this repository to send a private report. Include what you found, how to reproduce it and which files are affected.

## Scope

- `tools/capture_server.py` listens on `127.0.0.1` only and is meant to run on your own PC while you capture.
- `train.py` and `export.py` load saved weights with `weights_only=True`, so a weights file cannot run code.
- The benchmark downloads its reference model from a pinned Hugging Face commit and checks its SHA 256 before use.
