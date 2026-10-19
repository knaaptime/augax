.. _api_ref:

.. currentmodule:: augax

API reference
=============

The public API of ``augax`` is one sampler per augmentation scheme; each works
inside ``jax.jit``, ``jax.vmap``, ``lax.scan``, and ``jax.pmap``.

Sampling
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pg_sample

Internal helpers
~~~~~~~~~~~~~~~~

These functions support :func:`pg_sample` and are documented for completeness;
they are not part of the stable public API.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   _seed_from_key
   _dispatch
   _sample_batched
   _require_x64

Module contents
---------------

.. automodule:: augax
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
