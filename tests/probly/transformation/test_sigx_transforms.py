"""Tests for the helpers in ``probly.transformation._sigx_transforms``.

The module under test imports ``sigx_gen`` at module top level. ``sigx_gen``
is only present in the ``lint`` dependency group (it's used to generate type
stubs at lint time), so we skip the entire module when it isn't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sigx_gen")


class TestSigxTransformsHelpers:
    """Helpers in transformation/_sigx_transforms.py."""

    def test_explicit_predictor_types_with_class(self) -> None:
        from probly.transformation._sigx_transforms import _explicit_predictor_types  # noqa: PLC0415

        class Foo:
            pass

        assert _explicit_predictor_types(Foo) == (Foo,)

    def test_explicit_predictor_types_with_collection(self) -> None:
        from probly.transformation._sigx_transforms import _explicit_predictor_types  # noqa: PLC0415

        class Foo:
            pass

        class Bar:
            pass

        # Tuple, list and set inputs should all be unwrapped.
        for col in [(Foo, Bar), [Foo, Bar], {Foo, Bar}, frozenset({Foo, Bar})]:
            res = _explicit_predictor_types(col)
            assert set(res) == {Foo, Bar}

    def test_explicit_predictor_types_drops_non_class_entries(self) -> None:
        from probly.transformation._sigx_transforms import _explicit_predictor_types  # noqa: PLC0415

        class Foo:
            pass

        # Mixed list with non-class entries -> only class entries returned.
        res = _explicit_predictor_types([Foo, "not-a-class", 42, None])
        assert res == (Foo,)

    def test_explicit_predictor_types_with_unknown_returns_empty(self) -> None:
        from probly.transformation._sigx_transforms import _explicit_predictor_types  # noqa: PLC0415

        # Non-class, non-iterable inputs return an empty tuple.
        assert _explicit_predictor_types("hello") == ()
        assert _explicit_predictor_types(42) == ()
        assert _explicit_predictor_types(None) == ()

    def test_predictor_type_ref_in_probly_predictor_module(self) -> None:
        from probly.predictor._common import LogitDistributionPredictor  # noqa: PLC0415
        from probly.transformation._sigx_transforms import _predictor_type_ref  # noqa: PLC0415

        ref = _predictor_type_ref(LogitDistributionPredictor)
        assert ref.startswith("probly.predictor.")
        assert ref.endswith("LogitDistributionPredictor")

    def test_predictor_type_ref_outside_probly_predictor(self) -> None:
        from probly.transformation._sigx_transforms import _predictor_type_ref  # noqa: PLC0415

        class MyClass:
            pass

        ref = _predictor_type_ref(MyClass)
        # Returns module + qualname for non-probly-predictor classes.
        assert ref.startswith(MyClass.__module__)
        assert "MyClass" in ref

    def test_build_registry_names_by_type_returns_mapping(self) -> None:
        from probly.transformation._sigx_transforms import _build_registry_names_by_type  # noqa: PLC0415

        mapping = _build_registry_names_by_type()
        # The mapping must include at least one registered predictor.
        assert len(mapping) >= 1
        # Each key is a type, each value is a tuple of names.
        for typ, names in mapping.items():
            assert isinstance(typ, type)
            assert isinstance(names, tuple)
            assert all(isinstance(n, str) for n in names)


class TestSigxTransformsContextHelpers:
    """`_predictor_type_annotation_from_context` and `predictor_transformation_transform`.

    These functions need a TransformFactoryContext, which we mock with a
    minimal SimpleNamespace that provides the expected attributes.
    """

    def _make_ctx(self, permitted: object) -> object:
        """Build a minimal mock TransformFactoryContext."""
        from types import SimpleNamespace  # noqa: PLC0415

        bound_factory_args = SimpleNamespace(
            arguments={"permitted_predictor_types": permitted},
        )
        return SimpleNamespace(bound_factory_args=bound_factory_args)

    def test_annotation_with_no_permitted_types_returns_broad(self) -> None:
        from probly.transformation._sigx_transforms import (  # noqa: PLC0415
            _BROAD_PREDICTOR_TYPE_ANNOTATION,
            _predictor_type_annotation_from_context,
        )

        ctx = self._make_ctx(None)
        ann = _predictor_type_annotation_from_context(ctx)
        assert ann == _BROAD_PREDICTOR_TYPE_ANNOTATION

    def test_annotation_with_empty_iterable_returns_broad(self) -> None:
        from probly.transformation._sigx_transforms import (  # noqa: PLC0415
            _BROAD_PREDICTOR_TYPE_ANNOTATION,
            _predictor_type_annotation_from_context,
        )

        ctx = self._make_ctx([])
        ann = _predictor_type_annotation_from_context(ctx)
        # An empty list yields no explicit types -> broad annotation.
        assert ann == _BROAD_PREDICTOR_TYPE_ANNOTATION

    def test_annotation_with_explicit_logit_classifier(self) -> None:
        from probly.predictor import LogitClassifier  # noqa: PLC0415
        from probly.transformation._sigx_transforms import _predictor_type_annotation_from_context  # noqa: PLC0415

        ctx = self._make_ctx((LogitClassifier,))
        ann = _predictor_type_annotation_from_context(ctx)
        # The annotation should include a Literal[...] section with the registry names
        # for LogitClassifier and end with `| None`.
        assert "Literal[" in ann
        assert "logit_classifier" in ann or "LogitDistributionPredictor" in ann
        assert ann.endswith("None")

    def test_predictor_transformation_transform_adds_kwarg(self) -> None:
        """Mock-based test of ``predictor_transformation_transform``.

        We construct a SignatureIR for a trivial function and run the transform;
        the result should expose a new kw-only argument named ``predictor_type``.
        """
        from probly.transformation._sigx_transforms import predictor_transformation_transform  # noqa: PLC0415

        # Build a minimal SignatureIR for a function with one positional arg.
        def _example(x: int) -> int:
            return x

        from sigx_gen.builder import SignatureBuilder as Builder  # alias  # noqa: PLC0415

        original_sig = (
            Builder.from_signature_object(_example).build()
            if hasattr(Builder, "from_signature_object")
            else Builder.from_callable(_example).build()
            if hasattr(Builder, "from_callable")
            else None
        )

        if original_sig is None:
            pytest.skip("Cannot construct SignatureIR with available SignatureBuilder API")

        from types import SimpleNamespace  # noqa: PLC0415

        ctx = SimpleNamespace(
            original=original_sig,
            bound_factory_args=SimpleNamespace(arguments={"permitted_predictor_types": None}),
        )
        result = predictor_transformation_transform(ctx)
        # The transformed signature should include a predictor_type parameter.
        assert any(p.name == "predictor_type" for p in result.params)
