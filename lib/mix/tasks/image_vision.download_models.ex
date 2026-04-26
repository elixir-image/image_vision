defmodule Mix.Tasks.ImageVision.DownloadModels do
  @shortdoc "Pre-downloads ImageVision default models into the local cache"

  @moduledoc """
  Pre-downloads the default models used by `Image.Classification`,
  `Image.Segmentation`, `Image.Detection`, `Image.Background`,
  `Image.Captioning`, and `Image.ZeroShot` so that first-call latency
  is eliminated and the application can run fully offline.

  By default, every category's models are fetched. Pass one or more
  of `--classify`, `--segment`, `--detect`, `--background`,
  `--caption`, `--zero-shot` to limit the scope. Any category whose
  optional dependency is not loaded is skipped with a notice rather
  than treated as an error.

  ONNX weights for segmentation and detection are stored under the
  `ImageVision.ModelCache` cache root (see that module for cache
  configuration). Bumblebee classifier and embedder weights are
  stored under Bumblebee's own HuggingFace cache (controlled by
  `BUMBLEBEE_CACHE_DIR` / the standard HF cache env vars).

  ## Usage

      mix image_vision.download_models
      mix image_vision.download_models --classify
      mix image_vision.download_models --segment --detect
      mix image_vision.download_models --background --caption --zero-shot

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

  * `--background` downloads the BiRefNet lite ONNX weights for
    background removal. Requires `:ortex`.

  * `--caption` downloads the BLIP image-captioning model, featurizer,
    tokenizer, and generation config. Requires `:bumblebee`.

  * `--zero-shot` downloads the CLIP model, featurizer, and tokenizer
    for zero-shot classification. Requires `:bumblebee`.

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

  @birefnet_repo "onnx-community/BiRefNet_lite-ONNX"
  @birefnet_files ["onnx/model.onnx"]

  @default_classifier_model {:hf, "facebook/convnext-tiny-224"}
  @default_classifier_featurizer {:hf, "facebook/convnext-tiny-224"}
  @default_embedder_model {:hf, "facebook/dinov2-base"}
  @default_embedder_featurizer {:hf, "facebook/dinov2-base"}
  @default_captioner_repo "Salesforce/blip-image-captioning-base"
  @default_zero_shot_repo "openai/clip-vit-base-patch32"

  @switches [
    classify: :boolean,
    segment: :boolean,
    detect: :boolean,
    background: :boolean,
    caption: :boolean,
    zero_shot: :boolean
  ]

  @impl Mix.Task
  def run(argv) do
    {options, _args, _invalid} = OptionParser.parse(argv, strict: @switches)

    categories =
      case Enum.filter(options, fn {_k, v} -> v end) do
        [] -> [:classify, :segment, :detect, :background, :caption, :zero_shot]
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

  defp download(:background) do
    Mix.shell().info("")
    Mix.shell().info("[background removal]")

    if ortex_loaded?() do
      Enum.each(@birefnet_files, &fetch_onnx(@birefnet_repo, &1))
    else
      Mix.shell().info("  skipped — :ortex dependency not loaded")
    end
  end

  defp download(:caption) do
    Mix.shell().info("")
    Mix.shell().info("[captioning]")

    if bumblebee_loaded?() do
      Application.ensure_all_started(:bumblebee)

      captioner = configuration(:captioner)
      repo = repo_from(captioner, :model, @default_captioner_repo)

      load_bumblebee(:model, {:hf, repo})
      load_bumblebee(:featurizer, {:hf, repo_from(captioner, :featurizer, repo)})
      load_bumblebee(:tokenizer, {:hf, repo_from(captioner, :tokenizer, repo)})
      load_bumblebee(:generation_config, {:hf, repo_from(captioner, :generation_config, repo)})
    else
      Mix.shell().info("  skipped — :bumblebee dependency not loaded")
    end
  end

  defp download(:zero_shot) do
    Mix.shell().info("")
    Mix.shell().info("[zero-shot classification]")

    if bumblebee_loaded?() do
      Application.ensure_all_started(:bumblebee)

      zero_shot = configuration(:zero_shot)
      repo = repo_from(zero_shot, :repo, @default_zero_shot_repo)

      load_bumblebee(:model, {:hf, repo})
      load_bumblebee(:featurizer, {:hf, repo})
      load_bumblebee(:tokenizer, {:hf, repo})
    else
      Mix.shell().info("  skipped — :bumblebee dependency not loaded")
    end
  end

  # Extract a HuggingFace repo string from a config keyword list.
  # Accepts either a `{:hf, "..."}` tuple or falls back to `default`.
  defp repo_from(config, key, default) do
    case Keyword.get(config, key) do
      {:hf, name} -> name
      _other -> default
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
        :tokenizer -> Bumblebee.load_tokenizer(spec)
        :generation_config -> Bumblebee.load_generation_config(spec)
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
