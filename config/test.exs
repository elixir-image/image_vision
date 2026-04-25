import Config

config :logger,
  level: :warning

config :nx, :default_defn_options, [compiler: EXLA]
