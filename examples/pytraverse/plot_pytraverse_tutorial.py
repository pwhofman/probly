"""==================================
A Brief Introduction to PyTraverse
==================================

The goal of this example is to showcase the key aspects of PyTraverse and how
they can be used to rewrite datastructures.

PyTraverse abstracts away recursion so that you can focus on the semantics of
the task at hand. It provides composable traversers that can be combined,
extended via single dispatch, and enriched with state variables.
"""

# %%
# Hello World Traverser
# ---------------------
# Let's start with the simplest possible traverser: one that does not actually
# traverse anything:
from __future__ import annotations

import pytraverse as t

# %%
# The core function offered by the ``pytraverse`` module is the
# :func:`~pytraverse.traverse` function. It takes an object to traverse and a
# traverser that should be applied to it. Here, the traverser just calls the
# ``.upper()`` method of the passed-in string.
#
# Note the ``@t.traverser`` decorator. It converts our simple object-processing
# function into a traverser that can be used with the ``traverse`` function.


@t.traverser
def my_traverser(s: str) -> str:
    return s.upper()


s = "hello world"
print(t.traverse(s, my_traverser))

# %%
# Down to Business
# ----------------
# The previous example was maybe a bit boring. After all, we did not traverse
# anything. Let's change that now...
#
# Consider the problem of computing the sum of all the integers in a (nested)
# datastructure:

data = [[1, 2, [3, [4, 5], 6], 7, 8], 9]
# We want to compute 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 45 here...

# %%
# The main idea behind the traverser library is to abstract away the recursion,
# so that you can focus on the semantics of the task at hand.
#
# Let's start by solving this task using a standard recursive implementation:


def recursive_sum(x: object) -> int:
    if isinstance(x, list):
        s = sum([recursive_sum(item) for item in x])
        return s
    return x


print(recursive_sum(data))

# %%
# That was easy enough. Now, let's rewrite this to use the pytraverse library:

from collections.abc import Callable


@t.traverser
def sum_traverser(x: object, traverse: Callable[[object], int]) -> int:
    if isinstance(x, list):
        s = sum([traverse(item) for item in x])
        return s
    return x


print(t.traverse(data, sum_traverser))

# %%
# *Not so different...* Indeed, the code is mostly identical to the previous
# one. The main difference is that the function no longer explicitly calls
# itself. Instead it receives a ``traverse`` function, which facilitates the
# recursive calls.
#
# For now, this second implementation is not any simpler than the first one. But
# the power of traversers becomes apparent when we start composing them.

# %%
# Let's Compose
# -------------
# In the spirit of separation of concerns, it might not always be desirable (or
# possible) to write a single large recursive function which processes your data
# to your heart's content.
#
# Instead, one might want to decouple the general logic of traversing (what are
# the children of a node?) from the actual processing logic (what do we do at
# each node?).
#
# Consider the following monolithic approach:


def mashed_multiply(x: object) -> object:
    if isinstance(x, list):
        return [mashed_multiply(item) for item in x]
    return x * 2


def mashed_add(x: object) -> object:
    if isinstance(x, list):
        return [mashed_add(item) for item in x]
    return x + 1


def mashed_exp(x: object) -> object:
    if isinstance(x, list):
        return [mashed_exp(item) for item in x]
    return x**2


print("Data:", data)
print("Multiply:", mashed_multiply(data))
print("Add:", mashed_add(data))
print("Exp:", mashed_exp(data))

# %%
# Note the redundant code in ``mashed_multiply``, ``mashed_add`` and
# ``mashed_exp``. For complicated traversal logic, monolithic recursive
# processors lead to duplicated, non-extensible and brittle code. Let's fix
# this using traversers:


@t.singledispatch_traverser
def data_traverser(x: list, traverse: Callable[[object], object]) -> list:
    return [traverse(item) for item in x]


@t.singledispatch_traverser
def mul_traverser(x: int) -> int:
    return x * 2


@t.singledispatch_traverser
def add_traverser(x: int) -> int:
    return x + 1


@t.singledispatch_traverser
def exp_traverser(x: int) -> int:
    return x**2


print("Data:", data)
print("Mul:", t.traverse(data, t.sequential(data_traverser, mul_traverser)))
print("Add:", t.traverse(data, t.sequential(data_traverser, add_traverser)))
print("Exp:", t.traverse(data, t.sequential(data_traverser, exp_traverser)))

# %%
# Here, two new functions are used:
#
# 1. ``@t.singledispatch_traverser`` is used instead of ``@t.traverser``.
# 2. ``t.sequential`` is used to combine traversers.
#
# The ``singledispatch_traverser`` decorator creates traversers that dispatch
# based on the type of the first argument. The ``sequential`` function composes
# multiple traversers into one that applies them in order.

# %%
# Multiple Dispatchers
# --------------------
# Singledispatch and sequential composition are already fairly powerful, but
# there is more to the story. What happens when our data contains types that our
# traverser does not know about?

mixed_data = {"a": 1, "b": [2, 3], "c": {"d": 4, "e": 5}}

print("Mixed Data:", mixed_data)
print("Mul:", t.traverse(mixed_data, t.sequential(data_traverser, mul_traverser)))
print("Add:", t.traverse(mixed_data, t.sequential(data_traverser, add_traverser)))
print("Pow:", t.traverse(mixed_data, t.sequential(data_traverser, exp_traverser)))

# %%
# Since ``data_traverser`` only knows how to deal with lists, we cannot process
# ``mixed_data``. We could rewrite the ``data_traverser`` to also deal with
# ``dict`` inputs, but instead let's extend it via registration:


@data_traverser.register
def _(x: dict, traverse: Callable[[object], object]) -> dict:
    return {k: traverse(v) for k, v in x.items()}


print("Mixed Data:", mixed_data)
print("Mul:", t.traverse(mixed_data, t.sequential(data_traverser, mul_traverser)))
print("Add:", t.traverse(mixed_data, t.sequential(data_traverser, add_traverser)))
print("Pow:", t.traverse(mixed_data, t.sequential(data_traverser, exp_traverser)))

# %%
# All we had to do was ``register`` a new dispatch handler with the
# ``data_traverser`` (note the ``x: dict`` type hint which enables this!) and
# all existing users of that traverser benefit from the extended functionality.

# %%
# Learning to Count
# -----------------
# So far we just played around in functional wonderland where all traversers
# independently did their little thing without much care for the rest of the
# world. Let's introduce state.
#
# A ``GlobalVariable`` allows traversers to share mutable state across the
# entire traversal:

LEAF_COUNT = t.GlobalVariable[int]("LEAF_COUNT", default=0)


@data_traverser.register(object)
def leaf_count_traverser(state: t.State) -> t.State:
    state[LEAF_COUNT] += 1
    return state


traversed_data, state = t.traverse_with_state(
    mixed_data,
    t.sequential(data_traverser, mul_traverser),
)

print("Data:", mixed_data)
print("Traversed Data:", traversed_data)
print("Leaf Count:", state[LEAF_COUNT])

# %%
# First, we define the global ``int`` variable ``LEAF_COUNT``. Second, we
# register a new default dispatcher with the ``data_traverser``. If no other
# dispatcher matches (i.e. neither the previously defined ``list`` and ``dict``
# handlers), the new traverser is executed and increments the counter.
#
# We can also use state for conditional traversal:


def is_even_leaf(state: t.State) -> bool:
    return state[LEAF_COUNT] % 2 == 0


conditional_mul_traverser = t.traverser(mul_traverser, traverse_if=is_even_leaf)

traversed_data = t.traverse(
    mixed_data,
    t.sequential(data_traverser, conditional_mul_traverser),
)

print("Data:", mixed_data)
print("Conditionally Traversed Data:", traversed_data)

# %%
# To realize the conditional traversal, we make use of another neat feature of
# the ``traverser`` decorator: ``traverse_if``. This optional parameter can be
# used to disable the resulting traverser given some predicate. Complementary to
# ``traverse_if`` there is also an analogous ``skip_if`` parameter.

# %%
# Too Deep
# --------
# Even/odd counting is nice, but what about depth counting? Let's now only
# apply the multiply traverser to nodes that are not at the root level.
#
# A ``StackVariable`` works like a ``GlobalVariable``, but updates are only
# visible in downstream traverser calls, not upstream:

DEPTH_COUNT = t.StackVariable[int]("DEPTH_COUNT", default=-1)


@t.traverser
def depth_counter_traverser(state: t.State) -> t.State:
    state[DEPTH_COUNT] += 1
    print(f"DEPTH_COUNT = {state[DEPTH_COUNT]} at object {state.object}")
    return state


depth_conditional_mul_traverser = t.traverser(mul_traverser, traverse_if=lambda state: state[DEPTH_COUNT] > 0)

traversed_data = t.traverse(
    mixed_data,
    t.sequential(
        depth_counter_traverser,
        data_traverser,
        depth_conditional_mul_traverser,
    ),
)

print("Data:", mixed_data)
print("Traversed Data:", traversed_data)

# %%
# Note that the ordering of composed traversers matters! The
# ``depth_counter_traverser`` must be executed **before** the
# ``data_traverser``, so that the ``DEPTH_COUNT`` is already incremented when
# the recursive calls happen:

traversed_data = t.traverse(
    mixed_data,
    t.sequential(
        data_traverser,
        depth_counter_traverser,  # Depth count increased after data_traverser
        depth_conditional_mul_traverser,
    ),
)

print("Data:", mixed_data)
print("Traversed Data:", traversed_data)

# %%
# If the ``depth_counter_traverser`` is executed after the ``data_traverser``,
# the recursive data traversal will reach the leaf nodes before the
# ``DEPTH_COUNT`` variable is increased. This highlights the importance of
# ordering composed traversers carefully.
