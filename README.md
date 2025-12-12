# html4llms

Small helper to shrink noisy `outerHTML` copied from DevTools into something LLM-friendly:
- trims long attributes
- shortens SVGs
- hides script/style bodies
- pretty-prints structure

## Install
Build a wheel with `uv build`, then install the CLI locally (this creates the `html4llms` command):
```commandline
uv build
uv tool install --from . html4llms
```
Prefer to run without installing? You can keep using `uv run html4llms/cli.py ...`.

## Default usage (clipboard → clipboard)
Read HTML from the macOS clipboard (`pbpaste`), simplify it, then write the result back to the clipboard (`pbcopy`):
```commandline
html4llms
```
If clipboard tools are missing, it falls back to stdout.

## Other ways to run
- From stdin to stdout/file:
    ```
    pbpaste | html4llms > reduced.html
    ```
- From a file to a file:
    ```
    html4llms --in big.html --out reduced.html
    ```
- Tweak limits:
    ```
    html4llms --max-depth 18 --max-children 40
    ```

## Options
Key flags (see `html4llms --help` for more):
- `--in / --out` for file I/O
- `--max-depth`, `--max-children`, `--max-attr-len`, `--max-text-len`, `--max-classes`
- `--keep-style`, `--keep-data-attrs`
- `--pbpaste` to force clipboard input (default when stdin is a TTY)
