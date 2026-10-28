# V17 source baseline

- Created: 2026-10-28 05:04 PDT
- Read-only source: `/Users/huali/Desktop/RideResponse/rideshare_v26r_revision15d`
- V17 target: `/Users/huali/Desktop/RideResponse/rideshare_v26r_v17`
- Enclosing RideResponse Git HEAD at snapshot time: `de3f515c067a133ad93270352e8c5d9dfc67aeac`
- Included source files: 218
- Included tree size before manifest metadata: approximately 3.8 MiB
- File-hash manifest: `manifests/source_files.sha256`
- SHA-256 of file-hash manifest: `119a1411af7eb449a785abf4c48e355a64abbb0ab1586c8b13025372ec24624e`

## Copy policy

The snapshot was copied without modifying the Revision 15d source. The following generated or transient content was excluded:

- `runs/`
- `analysis/`
- `artifacts/`
- `.pytest_cache/`
- `__pycache__/`
- `*.pyc`, `*.pyo`
- `.DS_Store`
- `*.tmp`, `*.temp`

An `rsync -acn --delete` comparison using the same exclusions produced no differences after the copy.

## Provenance limitation

The Revision 15d directory is not tracked as files by the enclosing RideResponse Git repository; it appears as an untracked directory, so no pre-existing Git tag identifies this exact source tree. The content-addressed manifest above is therefore the authoritative source identity for the imported baseline. The V17 workspace will establish its own baseline commit and tag before any implementation patch is prepared.

## Scope

This baseline contains no V17 implementation changes. All V17 work-package changes must be prepared as discrete patches with one-page diff notes and require explicit user approval before application.
