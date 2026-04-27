import Config

config :nx,
  default_backend: EXLA.Backend,
  default_defn_options: [compiler: EXLA]

config :logger,
  level: :warning
