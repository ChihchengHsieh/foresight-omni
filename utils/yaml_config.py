"""YAML configuration support shared by command-line entry points."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


CONTROL_DESTINATIONS = {"config", "print_resolved_config", "set_values"}


def _flatten(
    node: Mapping[str, Any],
    path: tuple[str, ...] = (),
    values: Dict[str, Any] | None = None,
    origins: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Flatten organizational YAML sections to argparse destination names."""
    values = {} if values is None else values
    origins = {} if origins is None else origins

    for raw_key, value in node.items():
        if not isinstance(raw_key, str):
            raise ValueError("All YAML configuration keys must be strings.")
        key = raw_key.replace("-", "_")
        current_path = path + (raw_key,)

        if isinstance(value, Mapping):
            _flatten(value, current_path, values, origins)
            continue

        dotted = ".".join(current_path)
        if key in values:
            raise ValueError(
                f"Duplicate parameter '{key}' at '{origins[key]}' and '{dotted}'. "
                "Leaf names must be unique across YAML sections."
            )
        values[key] = value
        origins[key] = dotted

    return values


def _actions(parser: argparse.ArgumentParser) -> Dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions
        if action.dest not in {argparse.SUPPRESS, "help"}
    }


def _validate(values: Mapping[str, Any], parser: argparse.ArgumentParser) -> None:
    actions = _actions(parser)
    unknown = sorted(set(values) - set(actions))
    if unknown:
        raise ValueError("Unknown configuration parameter(s): " + ", ".join(unknown))

    for key, value in values.items():
        action = actions[key]
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                raise ValueError(f"Configuration parameter '{key}' must be boolean.")
        if action.choices is not None and value is not None and value not in action.choices:
            choices = ", ".join(map(str, action.choices))
            raise ValueError(
                f"Invalid value for '{key}': {value!r}. Choose from: {choices}."
            )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML experiment config; explicit CLI arguments take precedence.",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Highest-precedence override, e.g. --set early_stop=false.",
    )
    parser.add_argument(
        "--print_resolved_config",
        action="store_true",
        help="Print the final configuration before training.",
    )


def _config_path(argv: Iterable[str]) -> str | None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    known, _ = bootstrap.parse_known_args(list(argv))
    return known.config


def _load(path: str | None, parser: argparse.ArgumentParser) -> Dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ValueError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    document = {} if document is None else document
    if not isinstance(document, Mapping):
        raise ValueError("The YAML document root must be a mapping.")
    values = _flatten(document)
    _validate(values, parser)
    return values


def _apply_set(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    actions = _actions(parser)
    for assignment in args.set_values:
        if "=" not in assignment:
            parser.error(f"Invalid --set value {assignment!r}; expected KEY=VALUE.")
        raw_key, raw_value = assignment.split("=", 1)
        key = raw_key.strip().replace("-", "_")
        if key in CONTROL_DESTINATIONS or key not in actions:
            parser.error(f"Unknown --set parameter: {key}")
        try:
            value = yaml.safe_load(raw_value)
            _validate({key: value}, parser)
        except (ValueError, yaml.YAMLError) as exc:
            parser.error(str(exc))
        setattr(args, key, value)


def _record(args: argparse.Namespace) -> None:
    resolved = {
        key: value
        for key, value in sorted(vars(args).items())
        if key not in CONTROL_DESTINATIONS
    }
    rendered = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    if args.print_resolved_config:
        print(rendered, end="")
    if args.output_dir and int(os.environ.get("RANK", "0")) == 0:
        output_dir = Path(args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "resolved_config.yaml"
        temporary = output_dir / ".resolved_config.yaml.tmp"
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(target)
        print(f"Resolved configuration saved to {target}", file=sys.stderr)


def parse_configured_args(
    parser: argparse.ArgumentParser, argv: Iterable[str] | None = None
) -> argparse.Namespace:
    """Resolve defaults < YAML < CLI < --set and persist the result."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        parser.set_defaults(**_load(_config_path(argv), parser))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    args = parser.parse_args(argv)
    _apply_set(args, parser)
    _record(args)
    return args
