import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from .config import load_config, ProviderMode
from .frame_provider import FrameProvider
from .service import InferenceService


def main():
    parser = argparse.ArgumentParser(description="RGBT inference service (Hailo + WIRIS camera → ZMQ)")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to config YAML (default: config.yaml next to this package)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Treat SIGTERM (docker stop) the same as SIGINT so the finally block runs.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    config = load_config(args.config)

    if config.service.record_only:
        _run_record_only(config)
        return

    provider = _build_provider(config)
    service = InferenceService(config, provider)
    service.run()


def _run_record_only(config) -> None:
    from .frame_provider.live import LiveFrameProvider

    if config.provider.mode != ProviderMode.LIVE:
        print("record_only requires provider.mode = live", file=sys.stderr)
        sys.exit(1)
    if not (config.capture and config.capture.enabled):
        print("record_only requires capture.enabled = true", file=sys.stderr)
        sys.exit(1)

    default_sync_fps = config.service.max_fps or config.camera.fps
    max_sync_gap = config.service.max_sync_gap_s or (2.0 / default_sync_fps)
    provider = LiveFrameProvider(
        camera_cfg=config.camera,
        alignment_cfg=config.alignment,
        max_sync_gap_s=max_sync_gap,
        capture_cfg=config.capture,
    )

    print(f"Record-only mode — no inference", flush=True)
    print(provider.describe(), flush=True)
    print("Press Ctrl-C to stop.\n", flush=True)

    provider.start()
    stop = threading.Event()
    try:
        stop.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        provider.stop()


def _build_provider(config) -> FrameProvider:
    mode = config.provider.mode

    if mode == ProviderMode.LIVE:
        from .frame_provider.live import LiveFrameProvider
        default_sync_fps = config.service.max_fps or config.camera.fps
        max_sync_gap = config.service.max_sync_gap_s or (2.0 / default_sync_fps)
        return LiveFrameProvider(
            camera_cfg=config.camera,
            alignment_cfg=config.alignment,
            max_sync_gap_s=max_sync_gap,
            capture_cfg=config.capture,
        )

    if mode == ProviderMode.M3FD:
        from .frame_provider.m3fd import M3fdFrameProvider
        dataset = config.provider.dataset
        return M3fdFrameProvider(
            root=Path(dataset.root),
            splits=dataset.splits,
            fps=dataset.fps,
            loop=dataset.loop and dataset.max_frames is None,
            max_frames=dataset.max_frames,
        )

    if mode == ProviderMode.THREE_D_NET:
        from .frame_provider.three_d_net import ThreeDNetFrameProvider
        dataset = config.provider.dataset
        return ThreeDNetFrameProvider(
            root=Path(dataset.root),
            splits=dataset.splits,
            videos=dataset.videos,
            fps=dataset.fps,
            loop=dataset.loop and dataset.max_frames is None,
            max_frames=dataset.max_frames,
        )

    if mode == ProviderMode.RTDOD:
        from .frame_provider.rtdod import RtdodFrameProvider
        dataset = config.provider.dataset
        return RtdodFrameProvider(
            root=Path(dataset.root),
            splits=dataset.splits,
            fps=dataset.fps,
            loop=dataset.loop and dataset.max_frames is None,
            max_frames=dataset.max_frames,
        )

    raise ValueError(f"Unknown provider mode: {mode!r}")


if __name__ == "__main__":
    main()
