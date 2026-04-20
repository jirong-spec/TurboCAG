#!/usr/bin/env python3
"""migrate_store.py — Migrate a TurboCAG KV store to the new SHA-256 naming scheme.

Background
──────────
Old stores used  {md5(doc_id)[:12]}_L{layer:02d}_{scheme}.{bin,meta}
New stores use   {sha256(doc_id)[:16]}_L{layer:02d}_{scheme}.{bin,meta}
                 + manifest.json for integrity verification.

Because the MD5 prefix cannot be reversed, this script requires the caller to
supply the list of doc_ids that were packed into the store.

What it does
────────────
For each doc_id the script:
  1. Computes the old (MD5[:12]) and new (SHA256[:16]) filename prefixes.
  2. Finds all old-format .bin/.meta files matching the old prefix.
  3. Renames them to the new prefix (unless the target already exists).
  4. Adopts any already-new-format files that are not yet in the manifest.
  5. Writes an updated manifest.json.

Safety
──────
Default mode is a dry run — pass --yes to apply changes.
Files are renamed (not copied); the original paths no longer exist after
migration. Run with --yes only after verifying the dry-run output.

Usage
─────
    python scripts/migrate_store.py --store ./kv_store --doc-ids doc_ids.txt
    python scripts/migrate_store.py --store ./kv_store --doc-ids doc_ids.txt --yes

doc-ids file format (one per line; JSONL with "doc_id" field also accepted):
    my_document_1
    my_document_2
    ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


# ── hash helpers ──────────────────────────────────────────────────────────── #

def _md5_prefix(doc_id: str) -> str:
    return hashlib.md5(doc_id.encode()).hexdigest()[:12]

def _sha256_prefix(doc_id: str) -> str:
    return hashlib.sha256(doc_id.encode()).hexdigest()[:16]

def _sha256_full(doc_id: str) -> str:
    return hashlib.sha256(doc_id.encode()).hexdigest()


# ── doc-id loader ─────────────────────────────────────────────────────────── #

def _load_doc_ids(path: Path) -> list[str]:
    doc_ids: list[str] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                doc_ids.append(obj["doc_id"])
            except (json.JSONDecodeError, KeyError):
                doc_ids.append(line)
    return doc_ids


# ── manifest I/O ──────────────────────────────────────────────────────────── #

def _load_manifest(store_dir: Path) -> dict:
    p = store_dir / "manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"v": 1, "entries": {}}

def _save_manifest(store_dir: Path, manifest: dict, dry_run: bool) -> None:
    p = store_dir / "manifest.json"
    if dry_run:
        return
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(p)


# ── num_tokens helper ─────────────────────────────────────────────────────── #

def _read_num_tokens(meta_path: Path) -> int | None:
    if not meta_path.exists():
        return None
    try:
        import torch
        meta = torch.load(meta_path, weights_only=False)
        return int(meta.get("num_tokens", 0))
    except Exception:
        return None


# ── stem parser ───────────────────────────────────────────────────────────── #

def _parse_stem(stem: str) -> tuple[str, int, str] | None:
    """Parse '{prefix}_L{layer:02d}_{scheme}' → (prefix, layer, scheme) or None."""
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    layer_str = parts[1]
    if not (layer_str.startswith("L") and layer_str[1:].isdigit()):
        return None
    scheme = "_".join(parts[2:])  # handles turbo_prod, turbo_mse
    return parts[0], int(layer_str[1:]), scheme


# ── migration ─────────────────────────────────────────────────────────────── #

def migrate_store(store_dir: Path, doc_ids: list[str], dry_run: bool) -> None:
    manifest = _load_manifest(store_dir)
    entries = manifest.setdefault("entries", {})

    renamed = 0
    skipped = 0
    adopted = 0
    errors  = 0

    for doc_id in doc_ids:
        old_pfx  = _md5_prefix(doc_id)
        new_pfx  = _sha256_prefix(doc_id)
        full_hash = _sha256_full(doc_id)

        doc_entry = entries.setdefault(doc_id, {"sha256": full_hash, "layers": {}})
        doc_entry["sha256"] = full_hash

        # ── rename old-format files ──────────────────────────────── #
        for old_path in sorted(store_dir.glob(f"{old_pfx}_L*")):
            parsed = _parse_stem(old_path.stem)
            if parsed is None:
                continue
            _, layer, scheme = parsed

            new_stem = f"{new_pfx}_L{layer:02d}_{scheme}"
            new_path = old_path.with_name(new_stem + old_path.suffix)

            if new_path.exists():
                print(f"  SKIP    {old_path.name}  →  (target exists) {new_path.name}")
                skipped += 1
                continue

            action = "DRY-RENAME" if dry_run else "RENAME"
            print(f"  {action:10s} {old_path.name}  →  {new_path.name}")
            if not dry_run:
                try:
                    old_path.rename(new_path)
                except OSError as exc:
                    print(f"             ERROR: {exc}")
                    errors += 1
                    continue
            renamed += 1

            if old_path.suffix == ".bin":
                meta_path = new_path.with_suffix(".meta") if not dry_run \
                            else old_path.with_suffix(".meta")
                layer_key = f"{layer}::{scheme}"
                doc_entry["layers"][layer_key] = {
                    "key": new_stem,
                    "num_tokens": _read_num_tokens(meta_path),
                    "stored_at":  time.time(),
                }

        # ── adopt already-new-format files without manifest entries ─ #
        for new_path in sorted(store_dir.glob(f"{new_pfx}_L*")):
            if new_path.suffix != ".bin":
                continue
            parsed = _parse_stem(new_path.stem)
            if parsed is None:
                continue
            _, layer, scheme = parsed
            layer_key = f"{layer}::{scheme}"
            if layer_key in doc_entry["layers"]:
                continue  # already in manifest
            meta_path = new_path.with_suffix(".meta")
            doc_entry["layers"][layer_key] = {
                "key":        new_path.stem,
                "num_tokens": _read_num_tokens(meta_path),
                "stored_at":  time.time(),
            }
            print(f"  ADOPT      {new_path.name}")
            adopted += 1

    _save_manifest(store_dir, manifest, dry_run)

    print()
    if not dry_run:
        print(f"Manifest written → {store_dir / 'manifest.json'}")
    print(f"renamed={renamed}  skipped={skipped}  adopted={adopted}  errors={errors}")
    if dry_run:
        print("(dry run — no files were changed; pass --yes to apply)")


# ── CLI ───────────────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--store",    required=True, help="Path to the KV store directory")
    ap.add_argument("--doc-ids",  required=True, metavar="FILE",
                    help="File with doc_ids (one per line, or JSONL with 'doc_id' field)")
    ap.add_argument("--yes",      action="store_true",
                    help="Apply renames and write manifest (default: dry run)")
    args = ap.parse_args()

    store_dir = Path(args.store)
    if not store_dir.is_dir():
        ap.error(f"Store directory not found: {store_dir}")

    doc_ids_path = Path(args.doc_ids)
    if not doc_ids_path.exists():
        ap.error(f"doc-ids file not found: {doc_ids_path}")

    doc_ids = _load_doc_ids(doc_ids_path)
    if not doc_ids:
        ap.error("No doc_ids found in the provided file.")

    print(f"Store   : {store_dir.resolve()}")
    print(f"Doc IDs : {len(doc_ids)}")
    print(f"Mode    : {'LIVE — files will be renamed' if args.yes else 'DRY RUN (pass --yes to apply)'}")
    print()

    migrate_store(store_dir, doc_ids, dry_run=not args.yes)


if __name__ == "__main__":
    main()
