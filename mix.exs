defmodule ImageVision.MixProject do
  use Mix.Project

  @version "0.2.0"
  @app_name "image_vision"

  def project do
    [
      app: String.to_atom(@app_name),
      version: @version,
      elixir: "~> 1.17",
      deps: deps(),
      elixirc_paths: elixirc_paths(Mix.env()),
      source_url: "https://github.com/elixir-image/image_vision",
      docs: docs(),
      build_embedded: Mix.env() == :prod,
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      aliases: aliases(),
      dialyzer: [
        ignore_warnings: ".dialyzer_ignore_warnings",
        plt_add_apps: ~w(mix nx ortex bumblebee)a
      ],
      compilers: Mix.compilers()
    ]
  end

  defp description do
    """
    Simple image classification, segmentation, and object detection
    for the `image` library. Powered by Bumblebee (classification,
    embeddings) and Ortex (segmentation, detection) — strong defaults,
    no ML expertise required.
    """
  end

  def application do
    [
      mod: {ImageVision.Application, []},
      extra_applications: [:logger, :inets, :crypto]
    ]
  end

  defp deps do
    [
      # The core image-processing library.
      {:image, "~> 0.65"},

      # Used to download ONNX model weights on first call.
      {:req, "~> 0.5"},

      # Transitive workaround: the `:color` dep (pulled in via `:image`)
      # has a `Code.ensure_loaded?(Plug.Router)` check that misbehaves on
      # Elixir 1.20-rc.4. Including `:plug` ensures the conditional
      # branch compiles cleanly. Remove once `:color` upstream is fixed.
      {:plug, "~> 1.15", optional: true},

      # --- Optional ML deps ---
      #
      # These are declared `optional: true` so downstream consumers
      # must opt in explicitly. `image_vision` itself does not pull
      # them into the dependency graph, letting applications choose
      # their preferred backend and Bumblebee version.
      #
      # * Segmentation (`Image.Segmentation`) and object detection
      #   (`Image.Detection`) run ONNX models via Ortex.
      {:ortex, "~> 0.1", optional: true},
      #
      # * Classification (`Image.Classification`) and embedding
      #   (`Image.Classification.embed/2`) use Bumblebee servings.
      #   Users must separately add an Nx compiler to their own `mix.exs`.
      {:nx, "~> 0.11", optional: true, override: true},
      {:nx_image, "~> 0.1", optional: true},
      {:bumblebee, "~> 0.6", optional: true},
      # EXLA provides XLA-backed Nx.Defn compilation. Works on all
      # platforms including Apple Silicon (via XLA's CPU/NEON/AMX path).
      {:exla, "~> 0.11", optional: true},

      # --- Tooling ---
      {:ex_doc, "~> 0.18", only: [:release, :dev, :docs]},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp package do
    [
      maintainers: ["Kip Cole"],
      licenses: ["Apache-2.0"],
      links: links(),
      files: [
        "lib",
        "priv",
        "mix.exs",
        "README*",
        "CHANGELOG*",
        "LICENSE*"
      ]
    ]
  end

  def links do
    %{
      "GitHub" => "https://github.com/elixir-image/image_vision",
      "Readme" => "https://github.com/elixir-image/image_vision/blob/v#{@version}/README.md",
      "Changelog" => "https://github.com/elixir-image/image_vision/blob/v#{@version}/CHANGELOG.md",
      "image" => "https://hex.pm/packages/image",
      "libvips" => "https://www.libvips.org"
    }
  end

  def docs do
    [
      source_ref: "v#{@version}",
      main: "readme",
      extra_section: "Guides",
      extras: extras(),
      formatters: ["html"],
      skip_undefined_reference_warnings_on: ["changelog", "CHANGELOG.md"]
    ]
  end

  defp extras do
    Enum.filter(
      [
        "README.md",
        "LICENSE.md",
        "CHANGELOG.md",
        "guides/classification.md",
        "guides/segmentation.md",
        "guides/detection.md"
      ],
      &File.exists?/1
    )
  end

  def aliases do
    []
  end

  defp elixirc_paths(:test), do: ["lib", "test"]
  defp elixirc_paths(_), do: ["lib"]
end
