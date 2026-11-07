.. _api_ref:

.. currentmodule:: augax

API reference
=============

Each sampler works inside ``jax.jit``, ``jax.vmap``, ``lax.scan``, and ``jax.pmap``,
and has a ``*_numpy`` twin that makes the same draws for NumPy callers.

Samplers
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pg_sample
   ks_sample
   crt_sample
   gig_sample

NumPy samplers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pg_sample_numpy
   ks_sample_numpy
   crt_sample_numpy
   gig_sample_numpy

Pólya-Gamma moments
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pg_mean
   pg_var
