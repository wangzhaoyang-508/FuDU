# Contributing

Thanks for improving FuDU.

## Development

```bash
python -m pip install -e .
python examples/synthetic_stream.py
python -m unittest discover -s tests
```

## Pull Request Expectations

- Keep the detector-agnostic interface stable.
- Do not commit datasets, images, private paths, logs, or checkpoints.
- Add or update tests for changes to PGUQ, DeUE, fuzzy rules, or CLI behavior.
- Prefer small adapters over large detector framework forks.

## Reporting Issues

Please include:

- Python version and operating system.
- Exact command.
- A compact feature CSV and detection JSONL snippet when possible.
- Whether the issue is in PGUQ, DeUE, fuzzy sampling, or detector integration.
