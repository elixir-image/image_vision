import Config

config :logger,
  level: :warning

# Route all Nx.Defn computations (including Bumblebee featurizer
# preprocessing) through EXLA. This includes Apple Silicon — EXLA's
# XLA CPU path uses NEON/AMX and is significantly faster than the
# pure-Elixir Nx.Defn.Evaluator for both preprocessing and inference.
config :nx, :default_defn_options, [compiler: EXLA]
