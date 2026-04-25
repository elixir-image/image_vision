if ImageVision.bumblebee_configured?() do
  defmodule Image.Classification do
    @moduledoc """
    Image classification — what's in this image?

    Pass a `t:Vix.Vips.Image.t/0` to `classify/2` or `labels/2` and
    get back human-readable labels like `"sports car"` or
    `"Blenheim spaniel"`. Pass it to `embed/2` to get a fixed-size
    feature vector you can use for similarity search or downstream
    learning.

    ## Quick start

        iex> puppy = Image.open!("./test/support/images/puppy.webp")
        iex> [label | _rest] = Image.Classification.labels(puppy)
        iex> label
        "Blenheim spaniel"

    ## Default models

    The defaults are chosen for permissive licensing (Apache 2.0),
    reasonable size (<400 MB), and broad applicability:

    * **Classification** — `facebook/convnext-tiny-224`. ~110 MB,
      ~82.1% top-1 ImageNet, Apache 2.0. Returns one of 1000 ImageNet
      labels with a confidence score.

    * **Embedding** — `facebook/dinov2-base`. ~340 MB, Apache 2.0.
      Returns a 768-dim feature vector. Useful for "find similar
      images", clustering, or as input to a custom classifier.

    Power users can override every default through configuration or
    `classifier/1` options — see the configuration section below.

    ## Configuration

    Both classifier and embedder are configurable independently. The
    defaults are:

        # config/runtime.exs
        config :image_vision, :classifier,
          model: {:hf, "facebook/convnext-tiny-224"},
          featurizer: {:hf, "facebook/convnext-tiny-224"},
          model_options: [],
          featurizer_options: [],
          batch_size: 10,
          name: Image.Classification.Server,
          autostart: true

        config :image_vision, :embedder,
          model: {:hf, "facebook/dinov2-base"},
          featurizer: {:hf, "facebook/dinov2-base"},
          model_options: [],
          featurizer_options: [],
          batch_size: 10,
          name: Image.Classification.EmbeddingServer,
          autostart: false

    ## Servings and supervision

    Bumblebee servings are heavyweight processes — a model load can
    take several seconds and consume hundreds of megabytes. Each
    classification or embedding entry point runs against a named
    serving process so the model loads once and is reused.

    By default the classifier serving is autostarted by
    `ImageVision.Supervisor` when the `:image_vision` application
    starts. The embedding serving is not autostarted (most apps don't
    need it).

    To run a serving in your own supervision tree, set
    `autostart: false` and use `classifier/1` or `embedder/1` to get
    a child spec:

        # application.ex
        def start(_type, _args) do
          children = [
            Image.Classification.classifier(),
            Image.Classification.embedder(model: {:hf, "facebook/dinov2-large"})
          ]

          Supervisor.start_link(children, strategy: :one_for_one)
        end

    ## Optional dependency

    This module is only available when [Bumblebee](https://hex.pm/packages/bumblebee),
    [Nx](https://hex.pm/packages/nx), and an Nx compiler such as
    [EXLA](https://hex.pm/packages/exla) are configured in your
    application's `mix.exs`.

    """

    alias Vix.Vips.Image, as: Vimage

    @min_score 0.5

    @default_classifier [
      model: {:hf, "facebook/convnext-tiny-224"},
      featurizer: {:hf, "facebook/convnext-tiny-224"},
      model_options: [],
      featurizer_options: [],
      name: Image.Classification.Server,
      batch_size: 10,
      autostart: false
    ]

    @default_embedder [
      model: {:hf, "facebook/dinov2-base"},
      featurizer: {:hf, "facebook/dinov2-base"},
      model_options: [],
      featurizer_options: [],
      name: Image.Classification.EmbeddingServer,
      batch_size: 10,
      autostart: false
    ]

    @default_classifier_name @default_classifier[:name]
    @default_embedder_name @default_embedder[:name]

    @doc """
    Returns a child spec suitable for starting an image classification
    process as part of a supervision tree.

    ### Arguments

    * `configuration` is a keyword list merged over the default
      configuration.

    ### Options

    * `:model` is any image classification model supported by
      Bumblebee. The default is `{:hf, "facebook/convnext-tiny-224"}`.

    * `:featurizer` is any image featurizer supported by Bumblebee.
      The default is `{:hf, "facebook/convnext-tiny-224"}`.

    * `:model_options` is a keyword list of options passed to
      `Bumblebee.load_model/2`. The default is `[]`.

    * `:featurizer_options` is a keyword list of options passed to
      `Bumblebee.load_featurizer/2`. The default is `[]`.

    * `:name` is the name of the serving process. The default is
      `Image.Classification.Server`.

    * `:batch_size` is the maximum batch size, passed to
      `Bumblebee.Vision.image_classification/3`. The default is `10`.

    ### Returns

    * A child spec tuple suitable for `Supervisor.start_link/2`, or

    * `{:error, reason}` if the model could not be loaded.

    """
    @spec classifier(configuration :: Keyword.t()) ::
            {Nx.Serving, Keyword.t()} | {:error, Image.error()}
    def classifier(classifier \\ Application.get_env(:image_vision, :classifier, [])) do
      classifier = Keyword.merge(@default_classifier, classifier)

      model = Keyword.fetch!(classifier, :model)
      model_options = Keyword.fetch!(classifier, :model_options)

      featurizer = Keyword.fetch!(classifier, :featurizer)
      featurizer_options = Keyword.fetch!(classifier, :featurizer_options)

      batch_size = Keyword.fetch!(classifier, :batch_size)

      case Image.Classification.serving(
             model,
             model_options,
             featurizer,
             featurizer_options,
             batch_size
           ) do
        {:error, error} ->
          {:error, error}

        serving ->
          {Nx.Serving, serving: serving, name: classifier[:name], batch_timeout: 100}
      end
    end

    @doc """
    Returns a child spec suitable for starting an image embedding
    process as part of a supervision tree.

    Embeddings are fixed-size feature vectors useful for similarity
    search, clustering, or as input to a downstream classifier.

    ### Arguments

    * `configuration` is a keyword list merged over the default
      configuration.

    ### Options

    * `:model` is any image embedding model supported by Bumblebee.
      The default is `{:hf, "facebook/dinov2-base"}`.

    * `:featurizer` is any image featurizer supported by Bumblebee.
      The default is `{:hf, "facebook/dinov2-base"}`.

    * `:model_options` is a keyword list of options passed to
      `Bumblebee.load_model/2`. The default is `[]`.

    * `:featurizer_options` is a keyword list of options passed to
      `Bumblebee.load_featurizer/2`. The default is `[]`.

    * `:name` is the name of the serving process. The default is
      `Image.Classification.EmbeddingServer`.

    * `:batch_size` is the maximum batch size. The default is `10`.

    ### Returns

    * A child spec tuple suitable for `Supervisor.start_link/2`, or

    * `{:error, reason}` if the model could not be loaded.

    """
    @spec embedder(configuration :: Keyword.t()) ::
            {Nx.Serving, Keyword.t()} | {:error, Image.error()}
    def embedder(embedder \\ Application.get_env(:image_vision, :embedder, [])) do
      embedder = Keyword.merge(@default_embedder, embedder)

      model = Keyword.fetch!(embedder, :model)
      model_options = Keyword.fetch!(embedder, :model_options)

      featurizer = Keyword.fetch!(embedder, :featurizer)
      featurizer_options = Keyword.fetch!(embedder, :featurizer_options)

      batch_size = Keyword.fetch!(embedder, :batch_size)

      case embedding_serving(
             model,
             model_options,
             featurizer,
             featurizer_options,
             batch_size
           ) do
        {:error, error} ->
          {:error, error}

        serving ->
          {Nx.Serving, serving: serving, name: embedder[:name], batch_timeout: 100}
      end
    end

    @doc false
    def serving(model, model_options, featurizer, featurizer_options, batch_size) do
      with {:ok, model_info} <- Bumblebee.load_model(model, model_options),
           {:ok, featurizer} = Bumblebee.load_featurizer(featurizer, featurizer_options) do
        Bumblebee.Vision.image_classification(model_info, featurizer,
          compile: [batch_size: batch_size],
          defn_options: defn_options()
        )
      end
    end

    @doc false
    def embedding_serving(model, model_options, featurizer, featurizer_options, batch_size) do
      with {:ok, model_info} <- Bumblebee.load_model(model, model_options),
           {:ok, featurizer} = Bumblebee.load_featurizer(featurizer, featurizer_options) do
        Bumblebee.Vision.image_embedding(model_info, featurizer,
          compile: [batch_size: batch_size],
          defn_options: defn_options()
        )
      end
    end

    # Use EXLA as the Nx compiler when it is properly loaded and implements
    # the current Nx.Defn.Compiler protocol. Falls back to the default
    # evaluator when EXLA is absent or version-mismatched (e.g. EXLA 0.10
    # paired with Nx 0.11 does not export __compile__/4).
    defp defn_options do
      if Code.ensure_loaded?(EXLA) and function_exported?(EXLA, :__compile__, 4) do
        [compiler: EXLA]
      else
        []
      end
    end

    @doc """
    Classifies an image and returns the full prediction map.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:backend` is any valid Nx backend. The default is
      `Nx.default_backend/0`.

    * `:server` is the name of the serving process. The default is
      `Image.Classification.Server`.

    ### Returns

    * A map of the form `%{predictions: [%{label: String.t(), score: float()}]}`,
      or

    * `{:error, reason}`.

    ### Examples

        iex> puppy = Image.open!("./test/support/images/puppy.webp")
        iex> %{predictions: [%{label: _label, score: _score} | _rest]} =
        ...>   Image.Classification.classify(puppy)

    """
    @dialyzer {:nowarn_function, {:classify, 1}}
    @dialyzer {:nowarn_function, {:classify, 2}}

    @spec classify(image :: Vimage.t(), Keyword.t()) ::
            %{predictions: [%{label: String.t(), score: float()}]} | {:error, Image.error()}

    def classify(%Vimage{} = image, options \\ []) do
      backend = Keyword.get(options, :backend, Nx.default_backend())
      server = Keyword.get(options, :server, @default_classifier_name)

      with {:ok, tensor} <- to_classification_tensor(image, backend) do
        Nx.Serving.batched_run(server, tensor)
      end
    end

    @doc """
    Classifies an image and returns the labels that meet a minimum
    confidence score.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:backend` is any valid Nx backend. The default is
      `Nx.default_backend/0`.

    * `:min_score` is the minimum score, a float between `0.0` and
      `1.0`, that a label must meet to be returned. The default is
      `0.5`.

    * `:server` is the name of the serving process. The default is
      `Image.Classification.Server`.

    ### Returns

    * A list of labels. The list may be empty if no prediction meets
      `:min_score`.

    * `{:error, reason}`.

    ### Examples

        iex> car = Image.open!("./test/support/images/lamborghini-forsennato-concept.jpg")
        iex> Image.Classification.labels(car)
        ["sports car", "sport car"]

    """
    @dialyzer {:nowarn_function, {:labels, 1}}
    @dialyzer {:nowarn_function, {:labels, 2}}

    @spec labels(image :: Vimage.t(), options :: Keyword.t()) ::
            [String.t()] | {:error, Image.error()}

    def labels(%Vimage{} = image, options \\ []) do
      {min_score, options} = Keyword.pop(options, :min_score, @min_score)

      with %{predictions: predictions} <- classify(image, options) do
        predictions
        |> Enum.filter(fn %{score: score} -> score >= min_score end)
        |> Enum.flat_map(fn %{label: label} -> String.split(label, ", ") end)
      end
    end

    @doc """
    Computes a feature vector embedding of an image.

    Embeddings are fixed-size dense vectors. Two images with similar
    visual content will have similar embeddings, making this useful
    for similarity search, clustering, or as input to a downstream
    classifier.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:backend` is any valid Nx backend. The default is
      `Nx.default_backend/0`.

    * `:server` is the name of the embedding serving process. The
      default is `Image.Classification.EmbeddingServer`.

    ### Returns

    * An `Nx.Tensor` of shape `{embedding_size}` (e.g. `{768}` for
      DINOv2-base), or

    * `{:error, reason}`.

    ### Examples

        iex> puppy = Image.open!("./test/support/images/puppy.webp")
        iex> embedding = Image.Classification.embed(puppy)
        iex> Nx.shape(embedding)
        {768}

    """
    @dialyzer {:nowarn_function, {:embed, 1}}
    @dialyzer {:nowarn_function, {:embed, 2}}

    @spec embed(image :: Vimage.t(), options :: Keyword.t()) ::
            Nx.Tensor.t() | {:error, Image.error()}

    def embed(%Vimage{} = image, options \\ []) do
      backend = Keyword.get(options, :backend, Nx.default_backend())
      server = Keyword.get(options, :server, @default_embedder_name)

      with {:ok, tensor} <- to_classification_tensor(image, backend),
           %{embedding: embedding} <- Nx.Serving.batched_run(server, tensor) do
        embedding
      end
    end

    defp to_classification_tensor(%Vimage{} = image, backend) do
      with {:ok, flattened} <- Image.flatten(image),
           {:ok, srgb} <- Image.to_colorspace(flattened, :srgb) do
        Image.to_nx(srgb, shape: :hwc, backend: backend)
      end
    end
  end
end
