defmodule ImageVision do
  @moduledoc """
  `ImageVision` provides simple, opinionated image vision operations
  that sit alongside the [`image`](https://hex.pm/packages/image)
  library:

  * `Image.Classification` — what's in this image? (e.g. `"sports car"`,
    `"Blenheim spaniel"`). Powered by Bumblebee.

  * `Image.Segmentation` — which pixels belong to which object?
    Promptable ("click here to segment this thing") and class-labeled
    ("every pixel that's a road"). Powered by Ortex.

  * `Image.Detection` — where are the objects in this image, and what
    are they? Returns bounding boxes with class labels. Powered by
    Ortex.

  Each module loads strong, permissively-licensed default models on
  first call. Users do not have to know which model to pick, configure
  a backend, or manage a server — the defaults Just Work.

  ## Configuration

  ### Cache directory for downloaded ONNX models

  `Image.Segmentation` and `Image.Detection` download ONNX model
  weights from HuggingFace on first call and cache them on disk.
  The cache directory can be configured:

      config :image_vision, :cache_dir, "/path/to/cache"

  If unset, the default is `:filename.basedir(:user_cache, "image_vision")`
  (XDG-compliant per-user cache).

  ### Bumblebee servings (autostart)

  `Image.Classification` runs a supervised Bumblebee serving. It can be
  autostarted under `ImageVision.Supervisor` when the `:image_vision`
  application starts. Configure in `config/runtime.exs`:

      config :image_vision, :classifier,
        model: {:hf, "facebook/convnext-tiny-224"},
        featurizer: {:hf, "facebook/convnext-tiny-224"},
        autostart: true

  Setting `autostart: false` (the default) leaves the service unstarted
  — call `Image.Classification.classifier/1` to retrieve a child spec
  for your own supervision tree.

  ## Optional dependencies

  Each module compiles only when its underlying dependencies are
  present, so `:image_vision` costs nothing if you don't need it:

  * `Image.Classification` requires `:bumblebee`, `:nx`, and an Nx
    backend such as `:exla` at runtime.

  * `Image.Segmentation` and `Image.Detection` require `:ortex`.

  """

  @doc false
  def bumblebee_configured? do
    # Only Nx and Bumblebee are required at compile time for the
    # `Image.Classification` module to be defined. An Nx compiler
    # backend (`:exla`, `:torchx`, …) is also required at runtime
    # for the servings to actually run, but we don't gate compilation
    # on it — users will get a clear error from
    # `Application.ensure_all_started/1` at boot if it is missing.
    Enum.reduce_while([Nx, Bumblebee], true, fn mod, flag ->
      case Code.ensure_compiled(mod) do
        {:module, _module} -> {:cont, flag}
        _other -> {:halt, false}
      end
    end)
  end

  @doc false
  def ortex_configured? do
    match?({:module, _module}, Code.ensure_compiled(Ortex))
  end
end
