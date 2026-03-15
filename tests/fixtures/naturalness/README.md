This scaffold reserves three golden-set clip slots for phrase naturalness review.

Each clip directory should contain:
- `manifest.json`
- `expected_transcript.txt`
- `fillers.json`
- `listening_notes.md`
- source media referenced by `manifest.json`

The checked-in manifests and notes let the regression harness exist before the real
media is added to the repository or a private fixture store.
