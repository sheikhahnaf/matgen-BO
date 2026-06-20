#!/usr/bin/env python3
"""Download diffusion-models-for-materials review papers + alloy-capable
diffusion model papers, into data/diffusion_reviews/.

Borrows FORGE's full pipeline:
    - Elsevier: PII via Article Retrieval XML, then Object Retrieval for the
      real full PDF (Article Retrieval returns a 1-page stub — useless).
    - Springer: direct PDF URL (no auth).
    - Wiley:    TDM API with WILEY_TDM_TOKEN.
    - IOP / Nature / MDPI / Frontiers / TandF: direct publisher URLs.
    - Fallback: OpenAlex OA URL.

Credentials are sourced from /Volumes/SSD1_SMAAA/FORGE/.env (read-only).
All output goes inside our dedicated workspace; FORGE files are NOT modified.

Usage:
    python scripts/download_diffusion_reviews.py
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
import urllib.parse
import urllib.request

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "diffusion_reviews"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Read-only source for credentials
FORGE_ENV = Path("/Volumes/SSD1_SMAAA/FORGE/.env")


def _load_env():
    """Inject FORGE/.env into os.environ (only keys we use)."""
    if not FORGE_ENV.exists():
        return
    needed = {"ELSEVIER_API_KEY", "SPRINGER_META_API_KEY", "WILEY_TDM_TOKEN", "OPENALEX_EMAIL"}
    for line in FORGE_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k in needed and not os.environ.get(k):
            os.environ[k] = v


_load_env()

UA = f"MatinventHcapBO/1.0 (research; mailto:{os.environ.get('OPENALEX_EMAIL', 'researcher@tamu.edu')})"
ELSEVIER_API_KEY = os.environ.get("ELSEVIER_API_KEY", "")
WILEY_TDM_TOKEN = os.environ.get("WILEY_TDM_TOKEN", "")
TIMEOUT = 60
DELAY = 1.5

ARXIV_TARGETS = [
    ("2312.03687", "MatterGen — score-based diffusion crystal structures"),
    ("2511.03112", "MatInvent — RL fine-tuning of diffusion"),
    ("2504.00741", "Generative AI agents for inorganic materials"),
    ("2507.18314", "Atomistic Generative Diffusion for Materials"),
    ("2110.06197", "CDVAE — Crystal Diffusion VAE"),
    ("2202.02541", "DiffCSP — diffusion CSP"),
    ("2306.13196", "DiffCSP++ joint structure+composition"),
    ("2403.10846", "GNoME"),
    ("2202.13753", "ML-enabled HEA discovery (Rao 2022)"),
    ("2405.05303", "AdsorbDiff"),
    ("2410.01703", "Generative diffusion for amorphous"),
    ("2406.10538", "DPCDVAE property-conditional"),
    ("2403.02928", "Generative AI for materials review"),
    ("2402.06544", "Diffusion for inverse molecular design survey"),
    ("2502.12147", "eSEN"),
]

DOI_TARGETS = [
    ("10.1038/s41524-025-01901-1", "npj ComMatSci 2025 — generative diffusion amorphous"),
    ("10.1038/s41524-025-01930-w", "npj ComMatSci 2025 — PODGen conditional"),
    ("10.1038/s41586-025-08628-5", "Nature 2025 — MatterGen Zeni et al."),
    ("10.1038/s41563-025-02403-7", "Nature Materials 2025 — AI for materials design review"),
    ("10.1002/aidi.202500069",      "Adv. Intel. Discovery 2025 — Inverse design alloys via diffusion"),
    ("10.1021/jacs.5c14652",        "JACS 2025 — Diffusion bimetallic catalysts"),
    ("10.1126/science.abo4940",     "Science 2022 — ML-enabled HEA discovery (Rao)"),
    ("10.1038/s41524-024-01340-4",  "npj ComMatSci 2024 — review"),
]


def _http_get(url, headers=None, timeout=TIMEOUT):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status if hasattr(resp, "status") else resp.getcode(), resp.read()


def is_pdf(data: bytes) -> bool:
    return len(data) > 50_000 and data.startswith(b"%PDF-")


# ---------------------------------------------------------------------------
# Method-specific downloaders
# ---------------------------------------------------------------------------

def dl_arxiv(arxiv_id: str) -> bytes | None:
    try:
        _, data = _http_get(f"https://arxiv.org/pdf/{arxiv_id}")
        return data if is_pdf(data) else None
    except Exception as e:
        print(f"  arxiv-fail: {type(e).__name__}: {str(e)[:120]}")
        return None


def dl_elsevier(doi: str) -> bytes | None:
    """Elsevier: PII → Object Retrieval (the real full PDF, not the stub)."""
    if not ELSEVIER_API_KEY:
        return None
    try:
        # Step 1: get PII from article XML
        url = f"https://api.elsevier.com/content/article/doi/{urllib.parse.quote(doi)}"
        _, xml_b = _http_get(
            url,
            headers={
                "X-ELS-APIKey": ELSEVIER_API_KEY,
                "Accept": "text/xml",
                "User-Agent": UA,
            },
        )
        m = re.search(rb"<pii>([^<]+)</pii>", xml_b)
        if not m:
            print(f"  elsevier: no PII")
            return None
        pii = re.sub(r"[-() ]", "", m.group(1).decode())

        # Step 2: object retrieval for the real PDF
        url2 = f"https://api.elsevier.com/content/object/eid/1-s2.0-{pii}-main.pdf"
        _, pdf = _http_get(
            f"{url2}?httpAccept=application/pdf",
            headers={
                "X-ELS-APIKey": ELSEVIER_API_KEY,
                "User-Agent": UA,
            },
            timeout=120,
        )
        return pdf if is_pdf(pdf) else None
    except Exception as e:
        print(f"  elsevier-fail: {type(e).__name__}: {str(e)[:120]}")
        return None


def dl_springer(doi: str) -> bytes | None:
    try:
        _, data = _http_get(f"https://link.springer.com/content/pdf/{doi}.pdf")
        return data if is_pdf(data) else None
    except Exception as e:
        print(f"  springer-fail: {type(e).__name__}: {str(e)[:120]}")
        return None


def dl_wiley(doi: str) -> bytes | None:
    if not WILEY_TDM_TOKEN:
        return None
    try:
        _, data = _http_get(
            f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{urllib.parse.quote(doi, safe='')}",
            headers={
                "Wiley-TDM-Client-Token": WILEY_TDM_TOKEN,
                "User-Agent": UA,
            },
        )
        return data if is_pdf(data) else None
    except Exception as e:
        print(f"  wiley-fail: {type(e).__name__}: {str(e)[:120]}")
        return None


def dl_publisher_direct(doi: str) -> bytes | None:
    patterns = {
        "10.1088": lambda d: f"https://iopscience.iop.org/article/{d}/pdf",
        "10.1038": lambda d: f"https://www.nature.com/articles/{d.split('/')[-1]}.pdf",
        "10.3390": lambda d: f"https://www.mdpi.com/{'/'.join(d.split('/')[1:])}/pdf",
        "10.1080": lambda d: f"https://www.tandfonline.com/doi/pdf/{d}",
        "10.3389": lambda d: f"https://www.frontiersin.org/articles/{d}/pdf",
        "10.1126": lambda d: f"https://www.science.org/doi/pdf/{d}",  # AAAS / Science (often paywalled but worth trying)
        "10.1021": lambda d: f"https://pubs.acs.org/doi/pdf/{d}",      # ACS (usually paywalled)
    }
    for prefix, fn in patterns.items():
        if doi.startswith(prefix):
            try:
                _, data = _http_get(fn(doi))
                return data if is_pdf(data) else None
            except Exception as e:
                print(f"  direct-fail: {type(e).__name__}: {str(e)[:120]}")
                return None
    return None


def dl_openalex_oa(doi: str) -> bytes | None:
    try:
        _, body = _http_get(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}")
        meta = json.loads(body.decode())
        for key in (("primary_location", "pdf_url"),
                    ("best_oa_location", "pdf_url"),
                    ("open_access", "oa_url")):
            d = meta.get(key[0]) or {}
            url = d.get(key[1])
            if url:
                _, data = _http_get(url)
                if is_pdf(data):
                    return data
    except Exception as e:
        print(f"  openalex-fail: {type(e).__name__}: {str(e)[:120]}")
    return None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def download_doi(doi: str, label: str) -> bool:
    fname = OUT_DIR / f"doi_{doi.replace('/', '_').replace('.', '_')}.pdf"
    if fname.exists() and fname.stat().st_size > 50_000:
        print(f"[skip] {doi} (already downloaded)")
        return True

    methods = []
    if doi.startswith("10.1016"):
        methods.append(("elsevier", dl_elsevier))
    if doi.startswith("10.1007"):
        methods.append(("springer", dl_springer))
    if doi.startswith("10.1002") or doi.startswith("10.1111"):
        methods.append(("wiley", dl_wiley))
    methods.append(("direct", dl_publisher_direct))
    methods.append(("openalex", dl_openalex_oa))

    for name, fn in methods:
        data = fn(doi)
        if data and is_pdf(data):
            fname.write_bytes(data)
            print(f"[ok-{name:8s}] {doi}: {len(data)//1024} KB — {label[:55]}")
            return True

    print(f"[FAIL    ] {doi}: all methods failed — {label[:55]}")
    return False


def download_arxiv(arxiv_id: str, label: str) -> bool:
    fname = OUT_DIR / f"arxiv_{arxiv_id.replace('.', '_')}.pdf"
    if fname.exists() and fname.stat().st_size > 50_000:
        print(f"[skip] arxiv {arxiv_id}")
        return True
    data = dl_arxiv(arxiv_id)
    if data and is_pdf(data):
        fname.write_bytes(data)
        print(f"[ok-arxiv   ] {arxiv_id}: {len(data)//1024} KB — {label[:55]}")
        return True
    print(f"[FAIL    ] arxiv {arxiv_id}: {label[:55]}")
    return False


def main():
    print(f"[start] downloads → {OUT_DIR}")
    print(f"  Elsevier API: {'YES' if ELSEVIER_API_KEY else 'NO'}")
    print(f"  Wiley TDM:    {'YES' if WILEY_TDM_TOKEN else 'NO'}")
    n_ok = 0
    n_total = 0

    for arxiv_id, label in ARXIV_TARGETS:
        n_total += 1
        if download_arxiv(arxiv_id, label):
            n_ok += 1
        time.sleep(DELAY)

    for doi, label in DOI_TARGETS:
        n_total += 1
        if download_doi(doi, label):
            n_ok += 1
        time.sleep(DELAY)

    print(f"\n[done] {n_ok}/{n_total} downloaded → {OUT_DIR}")
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    with open(OUT_DIR / "INDEX.md", "w") as f:
        f.write("# Diffusion-for-materials review pool\n\n")
        f.write(f"Downloaded {len(pdfs)} PDFs.\n\n")
        for p in pdfs:
            f.write(f"- `{p.name}` ({p.stat().st_size//1024} KB)\n")


if __name__ == "__main__":
    main()
