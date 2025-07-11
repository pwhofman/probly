"""Generic functional datastructure traverser utilities."""

from . import composition, core, decorators, generic

## Core

Variable = core.Variable
GlobalVariable = core.GlobalVariable
StackVariable = core.StackVariable
ComputedVariable = core.ComputedVariable
computed = ComputedVariable  # Alias for convenience (intended to be used as a decorator)

type State[T] = core.State[T]
type TraverserResult[T] = core.TraverserResult[T]
type TraverserCallback[T] = core.TraverserCallback[T]
type Traverser[T] = core.Traverser[T]

traverse_with_state = core.traverse_with_state
traverse = core.traverse

## Traverser Decorator

traverser = decorators.traverser

## Composition

sequential = composition.sequential
top_sequential = composition.top_sequential
SingledispatchTraverser = composition.SingledispatchTraverser

singledispatch_traverser = composition.SingledispatchTraverser

## Generic traverser

generic_traverser = generic.generic_traverser
CLONE = generic.CLONE
TRAVERSE_KEYS = generic.TRAVERSE_KEYS
TRAVERSE_REVERSED = generic.TRAVERSE_REVERSED
