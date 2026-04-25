defmodule Mix.Tasks.ImageVision.Download do
  @shortdoc "Pre-downloads ImageVision default models into the local cache"

  @moduledoc """
  Pre-downloads the default models used by `Image.Classification`,
  `Image.Segmentation`, and `Image.Detection` so that first-call
  latency is eliminated and the application can run fully offline.

  By default, every category's models are fetched. Pass one or more of
  `--classify`, `--segment`, `--detect` to limit the scope. Any
  category whose optional dependency is not loaded is skipped with a
  notice rather than treated as an error.

  ONNX weights for segmentation and detection are stored under the
  `ImageVision.ModelCache` cache root (see that module for cache
  configuration). Bumblebee classifier and embedder weights are
  stored under Bumblebee's own HuggingFace cache (controlled by
  `BUMBLEBEE_CACHE_DIR` / the standard HF cache env vars).

  ## Usage

      mix image_vision.download
      mix image_vision.download --classify
      mix image_vision.download --segment --detect

  ## Configuration

  The task respects user overrides for the Bumblebee classifier and
  embedder:

      config :image_vision, :classifier,
        model: {:hf, "facebook/convnext-large-224-22k-1k"},
        featurizer: {:hf, "facebook/convnext-large-224-22k-1k"}

  Configured values are downloaded; unset values fall back to the
  library defaults.

  ## Options

  * `--classify` downloads the classifier and embedder Bumblebee
    models. Requires `:bumblebee` and `:nx`.

  * `--segment` downloads the SAM 2 and DETR-panoptic ONNX weights.
    Requires `:ortex`.

  * `--detect` downloads the RT-DETR ONNX weights. Requires
    `:ortex`.

  """

  use Mix.Task

  # Defaults mirror the runtime defaults in
  # `Image.Classification`, `Image.Segmentation`, and `Image.Detection`.
  # Kept here as the single source of truth for the download task —
  # the runtime modules carry their own copy because they are gated
  # on optional deps and may not be compiled.

  @sam_repo "SharpAI/sam2-hiera-tiny-onnx"
  @sam_files ["encoder.onnx", "decoder.onnx"]

  @detr_panoptic_repo "Xenova/detr-resnet-50-panoptic"
  @detr_panoptic_files ["onnx/model.onnx", "config.json"]

  @rtdetr_repo "onnx-community/rtdetr_r50vd"
  @rtdetr_files ["onnx/model.onnx"]

  @default_classifier_model {:hf, "facebook/convnext-tiny-224"}
  @default_classifier_featurizer {:hf, "facebook/convnext-tiny-224"}
  @default_embedder_model {:hf, "facebook/dinov2-base"}
  @default_embedder_featurizer {:hf, "facebook/dinov2-base"}

  @switches [classify: :boolean, segment: :boolean, detect: :boolean]

  @impl Mix.Task
  def run(argv) do
    {options, _args, _invalid} = OptionParser.parse(argv, strict: @switches)

    categories =
      case Enum.filter(options, fn {_k, v} -> v end) do
        [] -> [:classify, :segment, :detect]
        selected -> Enum.map(selected, fn {k, _} -> k end)
      end

    Mix.Task.run("app.config")
    Application.ensure_all_started(:req)

    Enum.each(categories, &download/1)

    Mix.shell().info("")
    Mix.shell().info("Done.")
  end

  defp download(:classify) do
    Mix.shell().info("")
    Mix.shell().info("[classification]")

    if bumblebee_loaded?() do
      Application.ensure_all_started(:bumblebee)

      classifier = configuration(:classifier)
      embedder = configuration(:embedder)

      load_bumblebee(:model, Keyword.get(classifier, :model, @default_classifier_model))

      load_bumblebee(
        :featurizer,
        Keyword.get(classifier, :featurizer, @default_classifier_featurizer)
      )

      load_bumblebee(:model, Keyword.get(embedder, :model, @default_embedder_model))

      load_bumblebee(
        :featurizer,
        Keyword.get(embedder, :featurizer, @default_embedder_featurizer)
      )
    else
      Mix.shell().info("  skipped — :bumblebee dependency not loaded")
    end
  end

  defp download(:segment) do
    Mix.shell().info("")
    Mix.shell().info("[segmentation]")

    if ortex_loaded?() do
      Enum.each(@sam_files, &fetch_onnx(@sam_repo, &1))
      Enum.each(@detr_panoptic_files, &fetch_onnx(@detr_panoptic_repo, &1))
    else
      Mix.shell().info("  skipped — :ortex dependency not loaded")
    end
  end

  defp download(:detect) do
    Mix.shell().info("")
    Mix.shell().info("[detection]")

    if ortex_loaded?() do
      Enum.each(@rtdetr_files, &fetch_onnx(@rtdetr_repo, &1))
    else
      Mix.shell().info("  skipped — :ortex dependency not loaded")
    end
  end

  defp fetch_onnx(repo, filename) do
    if ImageVision.ModelCache.cached?(repo, filename) do
      Mix.shell().info("  cached    #{repo}/#{filename}")
    else
      Mix.shell().info("  download  #{repo}/#{filename}")
      _path = ImageVision.ModelCache.fetch!(repo, filename)
      :ok
    end
  end

  defp load_bumblebee(kind, {:hf, name} = spec) do
    Mix.shell().info("  load      #{name} (#{kind})")

    result =
      case kind do
        :model -> Bumblebee.load_model(spec)
        :featurizer -> Bumblebee.load_featurizer(spec)
      end

    case result do
      {:ok, _loaded} ->
        :ok

      {:error, reason} ->
        Mix.raise("failed to load #{kind} #{inspect(spec)}: #{inspect(reason)}")
    end
  end

  defp configuration(key) do
    Application.get_env(:image_vision, key, [])
  end

  defp bumblebee_loaded? do
    Code.ensure_loaded?(Bumblebee)
  end

  defp ortex_loaded? do
    Code.ensure_loaded?(Ortex)
  end
end
