# Deploying the RegTrust Space

**NEVER run git inside this folder.** Deploy via the `huggingface_hub` Python
API, exactly as with the sibling Spaces (SAGETrust and others). The HF token
stays in a local `hf_token.txt` (or the `HF_TOKEN` env var) and is NEVER
committed.

## Files that must be uploaded
```
app.py
requirements.txt  README.md
src/model.py  src/trust.py  src/motif_shell.py
assets/checkpoints/fold_0.pt ... fold_4.pt
assets/JASPAR2026_CORE_vertebrates_non-redundant_pfms.txt
assets/ref_embeddings.npy
assets/calibration.json
assets/examples.json
```
(Do NOT upload hf_token.txt or DEPLOY.md.)

## First-time create + upload
```python
from huggingface_hub import HfApi, create_repo
token = open("hf_token.txt").read().strip()
repo = "<your-namespace>/regtrust"     # pick your namespace
create_repo(repo, repo_type="space", space_sdk="gradio", token=token, exist_ok=True)

api = HfApi(token=token)
api.upload_folder(
    folder_path=".",
    repo_id=repo,
    repo_type="space",
    ignore_patterns=["hf_token.txt", "DEPLOY.md", "__pycache__/*", "*.pyc"],
)
```

## Updating 1-3 files later (preferred over re-uploading everything)
```python
from huggingface_hub import HfApi
api = HfApi(token=open("hf_token.txt").read().strip())
api.upload_file(path_or_fileobj="app.py", path_in_repo="app.py",
                repo_id="<your-namespace>/regtrust", repo_type="space")
api.restart_space(repo_id="<your-namespace>/regtrust", factory_reboot=False)
```

## Notes
- `assets/` is the exact attribution checkpoint (fold_0, highest val_spearman)
  plus the full 5-model ensemble, the same calibrated cutoffs xai.py produced
  (`assets/calibration.json`), and the same seeded AD reference-pool draw
  `validate_trust_axes.py` uses -- this keeps the demo's numbers consistent
  with the manuscript by construction, not a separately retrained/recalibrated
  "deployment" model.
- To rebuild `assets/` after any pipeline change (retraining, recalibration):
  run `python precompute_app_assets.py` from the **project root** (needs the
  full local pipeline environment -- checkpoints, `hg38.2bit`,
  `phase1_sequence_windows.npz`, none of which live inside `hf_space/`), which
  regenerates everything under `hf_space/assets/`.
- `src/model.py`, `src/trust.py`, `src/motif_shell.py` are standalone copies of
  the project root's same-named files, with the `config.py` dependency removed
  (checkpoints carry their own architecture; the app has no need for the rest
  of `config.py`, which also pulls in `py2bit`/`hg38.2bit` this deployment
  does not ship). If the source project's model architecture or trust
  formulas change, re-sync these copies by hand -- there is no automated sync
  step.
- Local testing: `python app.py` from inside `hf_space/` (needs `src/` and
  `assets/` alongside it, exactly as uploaded). Requires `LD_LIBRARY_PATH`
  pointing at the conda env's own `lib/` on this sandbox machine (see
  project_status.md's "real environment issue hit while testing locally"
  note) -- irrelevant on HF's own container.
