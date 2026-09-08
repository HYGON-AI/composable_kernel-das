# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class TemplateInstance:
    path: Path
    template_name: str
    args: Tuple[str, ...]
    source: str


def split_template_args(args: str) -> List[str]:
    tokens = []
    start = 0
    angle_depth = 0
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0

    for index, char in enumerate(args):
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        elif (
            char == ","
            and angle_depth == 0
            and paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
        ):
            token = args[start:index].strip()
            if token:
                tokens.append(token)
            start = index + 1

    token = args[start:].strip()
    if token:
        tokens.append(token)
    return tokens


def extract_template_argument_strings(text: str, template_name: str) -> List[str]:
    results = []
    search_start = 0

    while True:
        name_start = text.find(template_name, search_start)
        if name_start == -1:
            return results

        open_index = name_start + len(template_name)
        while open_index < len(text) and text[open_index].isspace():
            open_index += 1

        if open_index >= len(text) or text[open_index] != "<":
            search_start = name_start + len(template_name)
            continue

        depth = 1
        index = open_index + 1
        while index < len(text):
            char = text[index]
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
                if depth == 0:
                    results.append(text[open_index + 1 : index])
                    search_start = index + 1
                    break
            index += 1
        else:
            raise ValueError(f"Unterminated template argument list for {template_name}")


def parse_template_instances_from_text(text: str, template_name: str, path: Path = Path("<memory>")) -> List[TemplateInstance]:
    return [
        TemplateInstance(
            path=path,
            template_name=template_name,
            args=tuple(split_template_args(arg_text)),
            source=arg_text,
        )
        for arg_text in extract_template_argument_strings(text, template_name)
    ]


def parse_template_instances_from_file(path: Path, template_name: str) -> List[TemplateInstance]:
    text = path.read_text(encoding="utf-8")
    return parse_template_instances_from_text(text, template_name, path=path)


def find_template_instances(
    root: Path,
    template_name: str,
    suffixes: Sequence[str] = (".cpp", ".hpp"),
) -> List[TemplateInstance]:
    root = Path(root)
    instances = []
    for path in sorted(_iter_source_files(root, suffixes)):
        instances.extend(parse_template_instances_from_file(path, template_name))
    return instances


def _iter_source_files(root: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    suffix_set = set(suffixes)
    if root.is_file():
        if root.suffix in suffix_set:
            yield root
        return

    for path in root.rglob("*"):
        if path.is_file() and path.suffix in suffix_set:
            yield path