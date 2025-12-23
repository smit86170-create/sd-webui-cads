Task: Create branch `cads-agents-md` in `smit86170-create/sd-webui-cads` and open a PR.

Add a new file `AGENTS.md` in the repo root with guidelines for working on this Automatic1111 WebUI extension:
- Keep compatibility with common A1111 versions.
- Do not add new dependencies beyond what WebUI already has.
- Keep infotext_fields and paste_field_names backward compatible.
- Prefer small, well-scoped commits.
- After changes, run: `python -m py_compile scripts/cads.py` and include output in PR description.

Deliverable: PR with only AGENTS.md.
