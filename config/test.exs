import Config

config :logger,
  level: :warning

# Route all Nx tensor allocations and Nx.Defn computations through
# EXLA. Without `default_backend`, only `defn`-compiled inference
# uses EXLA — the surrounding tensor work (image preprocessing,
# output reshaping, similarity dot products, etc.) falls back to
# the pure-Elixir Nx.BinaryBackend, which is orders of magnitude
# slower for image-sized tensors. This affected ML test runtime
# significantly before we set it.
config :nx,
  default_backend: EXLA.Backend,
  default_defn_options: [compiler: EXLA]
