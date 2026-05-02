"""
main.py — Entry point for the depth-based ground detection and path planning system.

    python main.py                     GUI mode (default)
    python main.py --headless          Terminal / headless mode
    python main.py --config path.toml  Use a custom config file
"""

from __future__ import annotations
import argparse


def _run_headless(config_path: str) -> None:
    import cv2
    from config import AppConfig
    from hardware import HardwareProfile
    from pipeline import DepthPipeline
    from visualization import overlay_ground, overlay_path, add_status_bar

    cfg = AppConfig.load(config_path)
    hw  = HardwareProfile.from_config(cfg)
    pipeline = DepthPipeline(cfg, hw)
    pipeline.load_calibration()
    pipeline.load_model()
    pipeline.start_capture()

    print("Press 'q' to quit, 's' to save outputs.")
    try:
        while not pipeline.is_stopped():
            result = pipeline.process_next_frame(timeout=0.5)
            if result is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            frame = result.rgb_frame
            colour_bgr = tuple(cfg.visualization.ground_colour_bgr)
            if result.ground_mask is not None:
                frame = overlay_ground(frame, result.ground_mask, colour_bgr,
                                       cfg.visualization.ground_overlay_alpha)
            if result.path is not None:
                frame = overlay_path(frame, result.path, result.start_point, result.end_point)
            add_status_bar(frame, result.plane)
            cv2.imshow(cfg.visualization.window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                pipeline.save_outputs(result)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth-based ground detection and path planning.")
    parser.add_argument("--config",   default="config.toml")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.headless:
        _run_headless(args.config)
    else:
        from gui import run
        run(args.config)


if __name__ == "__main__":
    main()
