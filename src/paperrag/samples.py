"""Official, reproducible downloads; no HTML-as-PDF or substitute corpus."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx
import pymupdf
from .storage import file_hash


def sample_manifest() -> list[dict]:
    source = Path(__file__).resolve().parents[2] / "examples" / "papers.json"
    if not source.is_file():
        source = Path(sys.prefix) / "share" / "paperrag" / "papers.json"
    return json.loads(source.read_text())


def validate_pdf(path: Path) -> None:
    with path.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise ValueError("Download is not a PDF (%PDF- header missing)")
    with pymupdf.open(path) as document:
        if document.needs_pass or not len(document):
            raise ValueError("Sample PDF is encrypted or has no pages")
        document[0].get_text()


def download_samples(destination: Path) -> dict:
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "samples.json"
    old = {p["paper_id"]: p for p in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    records, failures, results = [], [], []
    last_request = None
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "PaperRAG/0.1 local research sample downloader"}) as client:
        for paper in sample_manifest():
            path = destination / (paper["paper_id"] + ".pdf")
            tmp = path.with_suffix(".pdf.part")
            try:
                prior = old.get(paper["paper_id"])
                if path.exists():
                    validate_pdf(path)
                    digest = file_hash(path)
                    if prior and digest == prior["sha256"]:
                        records.append(prior)
                        results.append({"paper_id": paper["paper_id"], "status": "unchanged"})
                        continue
                    # Preserve any existing non-manifest file; don't overwrite user papers.
                    raise ValueError("Existing sample lacks matching download provenance; move it aside before downloading")
                if last_request is not None:
                    time.sleep(max(0, 3 - (time.monotonic() - last_request)))
                last_request = time.monotonic()
                with client.stream("GET", paper["url"]) as response:
                    response.raise_for_status()
                    final_url = str(response.url)
                    with tmp.open("wb") as f:
                        for block in response.iter_bytes(chunk_size=1024 * 1024):
                            f.write(block)
                validate_pdf(tmp)
                record = {"paper_id": paper["paper_id"], "title": paper["title"], "final_url": final_url,
                          "byte_count": tmp.stat().st_size, "downloaded_at": datetime.now(timezone.utc).isoformat(),
                          "sha256": file_hash(tmp)}
                tmp.replace(path)
                records.append(record)
                results.append({"paper_id": paper["paper_id"], "status": "downloaded", "bytes": record["byte_count"]})
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                failures.append({"paper_id": paper["paper_id"], "url": paper["url"], "error": f"{type(exc).__name__}: {exc}"})
                if paper["paper_id"] in old:
                    records.append(old[paper["paper_id"]])
            # Save each completed provenance record so a later interruption can resume safely.
            staging = manifest_path.with_suffix(".json.tmp")
            staging.write_text(json.dumps(records, indent=2) + "\n")
            staging.replace(manifest_path)
    return {"destination": str(destination), "papers": results, "failures": failures}
