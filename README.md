# Project

The project was tested with Python `3.12.2`, `3.11.9` and `3.10.11`.

## Building

To only run the Python code, create a virtual environment (recommended) with `python3 -m venv .venv` and activate it with `source .venv/bin/activate` (or in Windows `.\.venv\Scripts\activate`). Then install the requirements with `pip install -r requirements.txt`. Afterward, you can run the Python module with:

```bash
python -m project --all
```

This will run all the project parts (due to the `--all` flag) and save the plots under the [images directory](report/imgs).

### Configuration

To view all the available options run the command:

```bash
python -m project --help
```

For example, to only run labs 2 and 5, you can use the command:

```bash
python -m project --labs 2 5
```

To build the report you can pass the `--compile_pdf` flag, for example:

```bash
python -m project --compile_pdf --all
```
