#!/usr/bin/env python3
"""Local web server for the coronary branch labeler.

Run with: python server.py
Then open http://127.0.0.1:8765
"""
import base64
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
EXTRACTOR = ROOT / "extract_branch.py"
REQUIREMENTS = ROOT / "requirements.txt"
LABEL_OUTPUT_DIR = Path(os.environ.get("CORONARY_LABEL_OUTPUT_DIR", str(ROOT / "labels")))


def ensure_dependencies():
    """Install the extractor's Python dependencies on first launch."""
    required = ("numpy", "scipy", "skimage", "nibabel")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        return
    print("Installing required Python packages: " + ", ".join(missing), flush=True)
    command = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
        raise RuntimeError(
            "Could not install Python dependencies automatically. "
            "Check server.log for the pip error."
        )


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        route = urlparse(self.path).path
        if route == "/api/save-labels":
            self.save_labels()
            return
        if route != "/api/extract-branches":
            self.send_error(404, "Unknown API endpoint")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        boundary = content_type.split("boundary=", 1)[-1].strip().strip('"')
        if not boundary:
            self._json({"error": "Expected multipart/form-data with a mask field."}, 400)
            return
        marker = ("--" + boundary).encode()
        upload = None
        filename = "coronary_mask.nii.gz"
        for part in body.split(marker):
            if b"name=\"mask\"" not in part:
                continue
            head, sep, data = part.partition(b"\r\n\r\n")
            if not sep:
                continue
            for line in head.decode("utf-8", "ignore").split("\r\n"):
                if "filename=" in line:
                    filename = line.split("filename=", 1)[1].strip().strip('"')
            upload = data[:-2] if data.endswith(b"\r\n") else data
            break
        if not upload:
            self._json({"error": "No mask file was uploaded."}, 400)
            return

        try:
            with tempfile.TemporaryDirectory(prefix="coronary-labeler-") as temp:
                work = Path(temp)
                suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else Path(filename).suffix
                mask_path = work / ("uploaded_mask" + suffix)
                out_dir = work / "branches"
                mask_path.write_bytes(upload)
                interpreter = ["py", "-3"] if os.name == "nt" else [sys.executable]
                command = interpreter + [str(EXTRACTOR), str(mask_path), "--out-dir", str(out_dir)]
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
                if result.returncode != 0:
                    message = (result.stderr or result.stdout or "Branch extraction failed").strip()
                    print(f"Branch extraction failed: {message}", flush=True)
                    self._json({"error": message}, 422)
                    return
                files = sorted(out_dir.glob("*/branch_*_[lru].nii.gz"), key=self._branch_sort_key)
                if not files:
                    self._json({"error": "The extractor did not produce any branch masks."}, 422)
                    return
                branches = []
                for path in files:
                    name = path.name
                    side = "left" if name.rsplit("_", 1)[-1].startswith("l") else "right" if name.rsplit("_", 1)[-1].startswith("r") else "unsplit"
                    branches.append({
                        "name": name,
                        "side": side,
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    })
                self._json({"branches": branches, "log": result.stdout[-4000:]})
        except subprocess.TimeoutExpired:
            print("Branch extraction timed out after 5 minutes.", flush=True)
            self._json({"error": "Branch extraction timed out after 5 minutes."}, 504)
        except Exception as exc:
            print(f"Branch extraction server error: {exc}", flush=True)
            self._json({"error": str(exc)}, 500)

    def save_labels(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            case_id = str(payload.get("case_id", ""))
            import re
            if not re.fullmatch(r"CVD\d{1,40}", case_id):
                self._json({"error": "Invalid anonymised case ID."}, 400)
                return
            rows = payload.get("rows")
            if not isinstance(rows, list):
                self._json({"error": "Expected a rows array."}, 400)
                return
            fields = ["case_id", "branch_number", "cadrads_present", "stenosis", "cadrads_score", "plaque_burden", "case_notes", "labeled_at"]
            LABEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            destination = LABEL_OUTPUT_DIR / f"{case_id}_labels.csv"
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in fields})
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._json({"saved": True, "path": str(destination), "rows": len(rows)})
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    @staticmethod
    def _branch_sort_key(path):
        name = path.name
        side = name.rsplit("_", 1)[-1][0]
        try:
            number = int(name.split("_", 2)[1])
        except (IndexError, ValueError):
            number = 999999
        return ("lru".index(side) if side in "lru" else 9, number)

    def _json(self, payload, status=200):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    os.chdir(ROOT)
    ensure_dependencies()
    port = int(os.environ.get("CORONARY_LABEL_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Coronary branch labeler: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
