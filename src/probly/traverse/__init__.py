from typing import TypeAlias

from . import composition, core, decorators, generic

## Core

Variable = core.Variable
GlobalVariable = core.GlobalVariable
StackVariable = core.StackVariable
core.ComputedVariable = core.ComputedVariable
computed: TypeAlias = (
    core.ComputedVariable
)  # Alias for convenience (intended to be used as a decorator)

type State[T] = core.State[T]
type TraverserResult[T] = core.TraverserResult[T]
type TraverserCallback[T] = core.TraverserCallback[T]
type Traverser[T] = core.Traverser[T]

traverse = core.traverse

## Traverser Decorator

traverser = decorators.traverser

## Composition

sequential = composition.sequential
top_sequential = composition.top_sequential
singledispatch_traverser = composition.singledispatch_traverser

## Generic traverser

generic_traverser = generic.generic_traverser
CLONE = generic.CLONE
TRAVERSE_KEYS = generic.TRAVERSE_KEYS
