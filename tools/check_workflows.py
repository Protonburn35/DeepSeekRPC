#!/usr/bin/env python3
"""Parse every shipped workflow before CI execution (not a full Actions lint)."""
from pathlib import Path

import yaml


class UniqueLoader(yaml.BaseLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("workflow mapping has a duplicate or non-string key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / ".github/workflows").glob("*.yml"))
    for path in files:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
        if (not isinstance(document, dict) or not isinstance(document.get("on"), dict) or
                not isinstance(document.get("jobs"), dict) or not document["jobs"]):
            raise ValueError(f"invalid workflow document: {path.name}")
    print(f"Parsed {len(files)} workflow files with unique mappings; Actions semantics require actual CI.")


if __name__ == "__main__":
    main()
