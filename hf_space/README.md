---
title: RegTrust
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# RegTrust

A public demo of **RegTrust-XAI**: paste a DNA sequence and get a predicted
K562 ATAC-seq chromatin-accessibility score, plus the model's own trust
taxonomy (Scenario A-D: ensemble consensus x attribution coherence against a
real K562 transcription-factor motif panel) and an applicability-domain flag
(is this sequence anything like what the model was trained on).

K562 is ENCODE's gold-standard reference cell line for human gene-regulation
data (a CML-derived, BCR-ABL+ erythroleukemia line) -- this tool answers "how
accessible would this sequence be in K562 chromatin, and how much should you
trust that specific answer", not a general cross-tissue prediction. See the
app's own "About the method" tab for the full scope statement and the trust
taxonomy's validation numbers.

Code: https://github.com/MujeebOnawole/RegTrust-XAI
