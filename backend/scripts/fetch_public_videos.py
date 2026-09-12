# backend/scripts/fetch_public_videos.py
"""
Public Computer Vision & Surveillance Video Dataset Fetcher

Automates downloading standardized open-source computer vision benchmark videos
and facial surveillance test clips from well-known open-source repositories
(e.g., Intel IoT OpenVINO benchmarks, OpenCV official test suites, etc.) directly
into the datasets/videos/public_benchmarks directory.

These datasets provide unconstrained real-world test cases (walking crowds, varying
lighting, different camera angles, facial demographics) to supplement our synthetic
masked video streams when benchmarking video face detection and tracking.

Usage:
    python scripts/fetch_public_videos.py
    python scripts/fetch_public_videos.py --output-dir ../datasets/videos/public_benchmarks
"""

import argparse
import logging
import os
import ssl
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Curated list of high-quality public test video URLs for face detection, tracking, and surveillance
PUBLIC_VIDEO_DATASETS = [
    {
        "name": "intel_face_demographics_walking.mp4",
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4",
        "source": "Intel IoT OpenVINO Benchmark Suite",
        "description": "Person walking towards camera with clear facial views at varying distances."
    },
    {
        "name": "intel_face_demographics_pause.mp4",
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking-and-pause.mp4",
        "source": "Intel IoT OpenVINO Benchmark Suite",
        "description": "Individual walking and pausing before camera for recognition testing."
    },
    {
        "name": "intel_head_pose_female.mp4",
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female-and-male.mp4",
        "source": "Intel IoT OpenVINO Benchmark Suite",
        "description": "Head pose variations and face detection across multiple individuals."
    },
    {
        "name": "intel_classroom_monitoring.mp4",
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/classroom.mp4",
        "source": "Intel IoT OpenVINO Benchmark Suite",
        "description": "Multi-face classroom camera feed simulating wide-angle surveillance."
    },
    {
        "name": "intel_people_detection.mp4",
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4",
        "source": "Intel IoT OpenVINO Benchmark Suite",
        "description": "Crowded walkway people detection and trajectory tracking feed."
    },
    {
        "name": "opencv_surveillance_vtest.avi",
        "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
        "source": "OpenCV Official Benchmark Data",
        "description": "Standard pedestrian surveillance tracking benchmark video."
    },
    {
        "name": "opencv_character_megamind.avi",
        "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/Megamind.avi",
        "source": "OpenCV Official Benchmark Data",
        "description": "Standard character test video for tracking and facial extraction."
    },
    {
        "name": "opencv_optical_flow_tree.avi",
        "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/tree.avi",
        "source": "OpenCV Official Benchmark Data",
        "description": "Dynamic movement video clip for optical flow and tracker calibration."
    }
]


def download_file(url: str, target_path: Path) -> bool:
    """Downloads a file cleanly via HTTP/HTTPS with custom headers and error handling."""
    try:
        # Avoid SSL cert verification issues on local development environments if needed
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as response, open(target_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        return True
    except Exception as e:
        logging.warning(f"Failed to download {url}: {e}")
        if target_path.exists() and target_path.stat().st_size == 0:
            target_path.unlink(missing_ok=True)
        return False


def main():
    default_out = Path(__file__).resolve().parent.parent.parent / "datasets" / "videos" / "without_mask" / "public_benchmarks"

    parser = argparse.ArgumentParser(description="Fetch public open-source computer vision video datasets.")
    parser.add_argument("--output-dir", default=str(default_out), help="Target directory for downloaded video files")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting video dataset collection into {out_dir}...")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for item in PUBLIC_VIDEO_DATASETS:
        name = item["name"]
        url = item["url"]
        source = item["source"]
        desc = item["description"]
        target = out_dir / name

        if target.exists() and target.stat().st_size > 0 and not args.force:
            logging.info(f"[SKIP] {name} already exists ({target.stat().st_size / 1024:.1f} KB).")
            skip_count += 1
            continue

        logging.info(f"[DOWNLOAD] Fetching {name} from {source}...")
        ok = download_file(url, target)
        if ok and target.exists():
            size_kb = target.stat().st_size / 1024
            logging.info(f" -> Successfully downloaded {name} ({size_kb:.1f} KB): {desc}")
            success_count += 1
        else:
            logging.error(f" -> Could not retrieve {name} from {url}.")
            fail_count += 1

    total = len(PUBLIC_VIDEO_DATASETS)
    logging.info(f"Collection complete! Downloaded: {success_count} | Skipped (Existing): {skip_count} | Failed: {fail_count} out of {total} datasets.")


if __name__ == "__main__":
    main()
