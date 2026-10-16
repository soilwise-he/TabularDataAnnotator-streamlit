from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
PAGE_FILE = ROOT / "pages" / "01_📖_Input.py"


def _create_streamlit_stub() -> types.ModuleType:
    class _Dummy:
        def __call__(self, *args, **kwargs):
            return None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            return False
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    class _SessionState(dict):
        def get(self, key, default=None):
            return super().get(key, default)
        def pop(self, key, default=None):
            return super().pop(key, default) if key in self else default

    stub = types.ModuleType("streamlit")
    stub.session_state = _SessionState()
    stub.set_page_config = lambda *args, **kwargs: None
    stub.cache_data = lambda func=None, *args, **kwargs: (func if func else (lambda f: f))
    stub.cache_resource = lambda func=None, *args, **kwargs: (func if func else (lambda f: f))
    stub.error = lambda *args, **kwargs: None
    stub.warning = lambda *args, **kwargs: None
    stub.info = lambda *args, **kwargs: None
    stub.write = lambda *args, **kwargs: None
    stub.caption = lambda *args, **kwargs: None
    stub.markdown = lambda *args, **kwargs: None
    stub.text_input = lambda *args, **kwargs: ""
    stub.file_uploader = lambda *args, **kwargs: None
    stub.radio = lambda *args, **kwargs: kwargs.get("options", [None])[0] if kwargs.get("options") else None
    stub.empty = lambda *args, **kwargs: _Dummy()
    def _columns(*args, **kwargs):
        width = args[0] if args else 1
        if isinstance(width, (list, tuple)):
            count = len(width)
        else:
            count = int(width) if width is not None else 1
        return [_Dummy() for _ in range(count)]
    stub.columns = _columns
    stub.expander = lambda *args, **kwargs: _Dummy()
    stub.button = lambda *args, **kwargs: False
    stub.selectbox = lambda *args, **kwargs: kwargs.get("options", [None])[0] if kwargs.get("options") else None
    stub.dataframe = lambda *args, **kwargs: None
    stub.multiselect = lambda *args, **kwargs: []
    def _tabs(*args, **kwargs):
        first = args[0] if args else []
        count = len(first) if isinstance(first, (list, tuple)) else 0
        return [_Dummy() for _ in range(count)]
    stub.tabs = _tabs
    stub.logo = lambda *args, **kwargs: None
    stub.sidebar = _Dummy()
    stub.__getattr__ = lambda name: (lambda *args, **kwargs: None)
    return stub


def _install_streamlit_stub() -> types.ModuleType:
    stub = _create_streamlit_stub()
    existing = sys.modules.get("streamlit")
    if existing is not None:
        for name, value in vars(stub).items():
            if name.startswith("__"):
                continue
            setattr(existing, name, value)
        return existing
    sys.modules["streamlit"] = stub
    return stub


def load_app_helpers():
    """Load the existing app helper functions without reimplementing the metadata logic."""
    if not PAGE_FILE.exists():
        raise FileNotFoundError(f"Could not find app page: {PAGE_FILE}")

    _install_streamlit_stub()

    spec = importlib.util.spec_from_file_location("dataannot_input_page", PAGE_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


HELPERS = load_app_helpers()


def _normalize_input_links(raw_links: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
    for item in raw_links:
        value = str(item).strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        links.append(value)
    return links


def _read_link_list(path: str | Path) -> list[str]:
    repo_file = Path(path)
    if not repo_file.exists():
        raise FileNotFoundError(f"Link file not found: {repo_file}")
    lines = repo_file.read_text(encoding="utf-8").splitlines()
    return _normalize_input_links(lines)


def _safe_table_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "metadata_table"


def _download_bytes(url: str, timeout: tuple[int, int] = (10, 60), retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, timeout=timeout, stream=True) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        chunks.append(chunk)
                return b"".join(chunks)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < retries - 1:
                continue
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to download {url}")


def _process_zip_bytes(
    file_url: str,
    tabular_dict: dict[str, pd.DataFrame],
    context_files: list,
    filename_dict: dict[str, str],
    imported_metadata_by_filename: dict[str, pd.DataFrame],
) -> None:
    zip_bytes = _download_bytes(file_url)
    filename = HELPERS.filename_from_url(file_url)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in HELPERS.iter_interesting_zip_members(zf, [".csv", ".xlsx", ".xls", ".json"]):
            member_basename = info.filename.rsplit("/", 1)[-1]
            with zf.open(info) as fh:
                raw = fh.read()

            if member_basename.lower().endswith(".json"):
                parsed = HELPERS._try_parse_remote_metadata_bytes(raw, member_basename)
                if parsed:
                    imported_metadata_by_filename.update(parsed)
                continue

            if member_basename.lower().endswith(".csv"):
                if HELPERS._is_ignored_remote_csv_filename(member_basename):
                    continue
                if HELPERS._is_metadata_export_filename(member_basename):
                    parsed = HELPERS._parse_metadata_csv_bytes(raw, source_filename=member_basename, show_errors=False)
                    if parsed:
                        imported_metadata_by_filename.update(parsed)
                    continue
                df = HELPERS.read_csvBytes_with_sniffer(raw)
                tabular_dict[f"{member_basename} (from {filename})"] = df
                filename_dict[f"{member_basename} (from {filename})"] = file_url
                continue

            if member_basename.lower().endswith((".xlsx", ".xls")):
                bio = io.BytesIO(raw)
                bio.name = member_basename
                for sheet_name, df in HELPERS.get_excel(bio).items():
                    key = f"{member_basename} | {sheet_name} (from {filename})"
                    tabular_dict[key] = df
                    filename_dict[key] = file_url

        for info in HELPERS.iter_interesting_zip_members(zf, [".doc", ".docx", ".pdf", ".md", ".txt"]):
            with zf.open(info) as fh:
                raw = fh.read()
            context_files.append({
                "source_zip": filename,
                "member_path": info.filename,
                "bytes": raw,
            })


def collect_metadata_tables_for_zenodo_url(url: str) -> dict[str, pd.DataFrame]:
    """Collect metadata tables for one Zenodo record URL and return a dict keyed by table name."""
    record_id = HELPERS.get_record_id_from_Zenodo_url(url)
    tabular_dict: dict[str, pd.DataFrame] = {}
    filename_dict: dict[str, str] = {}
    zipped_context_files: list = []
    remote_imported_metadata_by_filename: dict[str, pd.DataFrame] = {}

    filtered_extensions_tabular = [".csv", ".xlsx", ".xls", ".json"]
    filtered_extensions_zip = [".zip"]

    files_url_tabular = HELPERS.get_files_URL_from_Zenodo_id(record_id, extensions=filtered_extensions_tabular)
    files_url_zip = HELPERS.get_files_URL_from_Zenodo_id(record_id, extensions=filtered_extensions_zip)

    for file_url in files_url_tabular:
        try:
            file_bytes = _download_bytes(file_url)
        except Exception:
            continue

        name_file = HELPERS.filename_from_url(file_url)
        ext_file = Path(name_file).suffix.lower()

        if ext_file == ".json":
            parsed = HELPERS._try_parse_remote_metadata_bytes(file_bytes, name_file)
            if parsed:
                remote_imported_metadata_by_filename.update(parsed)
            continue

        if ext_file == ".csv":
            if HELPERS._is_ignored_remote_csv_filename(name_file):
                continue
            if HELPERS._is_metadata_export_filename(name_file):
                parsed = HELPERS._parse_metadata_csv_bytes(file_bytes, source_filename=name_file, show_errors=False)
                if parsed:
                    remote_imported_metadata_by_filename.update(parsed)
                continue

            df = HELPERS.read_csvBytes_with_sniffer(file_bytes)
            tabular_dict[name_file] = df
            filename_dict[name_file] = file_url
            continue

        if ext_file in {".xlsx", ".xls"}:
            bio = io.BytesIO(file_bytes)
            bio.name = name_file
            for sheet_name, df in HELPERS.get_excel(bio).items():
                key = f"{name_file} | {sheet_name}"
                tabular_dict[key] = df
                filename_dict[key] = file_url

    for file_url in files_url_zip:
        try:
            _process_zip_bytes(file_url, tabular_dict, zipped_context_files, filename_dict, remote_imported_metadata_by_filename)
        except Exception:
            continue

    if not remote_imported_metadata_by_filename and not tabular_dict:
        raise ValueError(f"No metadata tables or tabular files discovered for record {record_id} ({url})")

    return remote_imported_metadata_by_filename or {name: df for name, df in tabular_dict.items()}


def export_record_tables(url: str, output_dir: Path) -> dict[str, pd.DataFrame]:
    record_id = HELPERS.get_record_id_from_Zenodo_url(url)
    metadata_tables = collect_metadata_tables_for_zenodo_url(url)
    record_dir = output_dir / str(record_id)
    record_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, str] = {}
    for table_name, df in metadata_tables.items():
        safe_name = _safe_table_name(table_name)
        file_path = record_dir / f"{safe_name}.csv"
        df.to_csv(file_path, index=False)
        summary[table_name] = str(file_path)

    manifest_path = record_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-export metadata tables from Zenodo record URLs.")
    parser.add_argument("--links", nargs="*", default=[], help="One or more Zenodo URLs to process.")
    parser.add_argument("--links-file", type=Path, default=None, help="Plain-text file with one Zenodo URL per line.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "zenodo_metadata_exports", help="Directory for CSV exports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = list(args.links)
    if args.links_file:
        urls.extend(_read_link_list(args.links_file))

    urls = _normalize_input_links(urls)
    if not urls:
        raise SystemExit("No Zenodo URLs provided. Use --links or --links-file.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, str]] = []
    for url in urls:
        record_id = HELPERS.get_record_id_from_Zenodo_url(url)
        try:
            tables = export_record_tables(url, output_dir)
            summary.append({
                "record_id": record_id,
                "url": url,
                "table_count": str(len(tables)),
                "tables": ", ".join(sorted(tables.keys())),
            })
            print(f"OK: {record_id} -> {len(tables)} metadata table(s)")
        except Exception as exc:
            print(f"ERROR: {record_id} -> {exc}")
            summary.append({
                "record_id": record_id,
                "url": url,
                "table_count": "0",
                "tables": "",
                "error": str(exc),
            })

    manifest_path = output_dir / "run_summary.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved run summary to: {manifest_path}")


if __name__ == "__main__":
    main()
