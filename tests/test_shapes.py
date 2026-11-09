"""Shape and mathematical invariant tests corresponding to `tests/test_shapes.py` in §5."""

from tests.test_shapes_and_contracts import (
    test_ast_registry_names_and_constructor_signatures_match_yamls,
    test_liskov_substitution_all_ablation_variants,
    test_schema_rejects_unknown_registry_or_invalid_constructor_args,
    test_yaml_inheritance_and_strict_schema,
)

__all__ = [
    "test_yaml_inheritance_and_strict_schema",
    "test_schema_rejects_unknown_registry_or_invalid_constructor_args",
    "test_ast_registry_names_and_constructor_signatures_match_yamls",
    "test_liskov_substitution_all_ablation_variants",
]
