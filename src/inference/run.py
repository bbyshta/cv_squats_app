"""Command line image inference entry point."""

from __future__ import annotations

import argparse

from src.utils.config import load_yaml_config
from src.utils.io import validate_image_path


NO_FINAL_MODEL_STATUS = "final_model_not_selected"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pose estimation on a single image.")
    parser.add_argument("--input", required=True, help="Path to a jpg, jpeg, or png image.")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="Path to application config. Default: configs/app.yaml",
    )
    parser.add_argument(
        "--rules-config",
        default="configs/pushup_rules.yaml",
        help="Path to push-up analysis rules config. Default: configs/pushup_rules.yaml",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        input_path = validate_image_path(args.input)
        config = load_yaml_config(args.config)
        load_yaml_config(args.rules_config)
        _validate_no_selected_backend(config)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    parser.exit(
        status=1,
        message=(
            "Error: no final pose model backend is selected yet. "
            "Complete the mandatory architecture comparison and configure the "
            "selected trained model before running image inference.\n"
            f"Input was validated: {input_path}\n"
        ),
    )


def _validate_no_selected_backend(config: dict) -> None:
    selected_model = config.get("selected_model")
    if not isinstance(selected_model, dict):
        raise ValueError("Config must contain selected_model mapping")

    status = selected_model.get("status")
    if status != NO_FINAL_MODEL_STATUS:
        raise ValueError(
            f"Config selected_model.status must be '{NO_FINAL_MODEL_STATUS}' "
            "until final model selection is completed"
        )
    if selected_model.get("backend") is not None:
        raise ValueError("Config selected_model.backend must stay null before final selection")


if __name__ == "__main__":
    main()
