if ImageVision.bumblebee_configured?() do
  defmodule Image.ZeroShot do
    @moduledoc """
    Zero-shot image classification — classify an image against
    arbitrary labels you supply at call time, without retraining.

    Where `Image.Classification` is constrained to whatever 1000
    ImageNet labels its model was trained on, `Image.ZeroShot` lets
    you provide your own set of candidate labels and asks the model
    "which of these best describes this image?". Powered by CLIP, a
    contrastive vision-language model.

    Three entry points cover different needs:

    * `classify/3` — return all candidate labels with scores, sorted
      descending. Best when you want to see the full distribution.

    * `label/3` — return just the single highest-scoring label.

    * `similarity/3` — compute cosine similarity between two images
      in CLIP's embedding space, useful for "find similar images".

    ## Quick start

        iex> _puppy = Image.open!("./test/support/images/puppy.webp")
        iex> # Image.ZeroShot.classify(puppy, ["a dog", "a cat", "a horse"])
        iex> # => [
        iex> #      %{label: "a dog", score: 0.97},
        iex> #      %{label: "a cat", score: 0.02},
        iex> #      %{label: "a horse", score: 0.01}
        iex> #    ]

    ## Default model

    [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
    — MIT licensed, ~600 MB. The original CLIP, broad training
    coverage and well-validated zero-shot behaviour. Override via the
    `:repo` option to use a larger CLIP variant.

    ## Prompt templates

    CLIP's accuracy on short labels improves significantly when each
    label is wrapped in a natural-language sentence. The default
    template is `"a photo of {label}"`, applied to every label before
    tokenisation. Override with the `:template` option:

        Image.ZeroShot.classify(image, ["dog", "cat"],
          template: "a close-up photograph of {label}")

    Pass `template: nil` to disable the template entirely (useful if
    your labels are already full sentences).

    ## Optional dependency

    This module is only available when [Bumblebee](https://hex.pm/packages/bumblebee),
    [Nx](https://hex.pm/packages/nx), and an Nx compiler such as
    [EXLA](https://hex.pm/packages/exla) are configured in your
    application's `mix.exs`.

    """

    alias Vix.Vips.Image, as: Vimage

    @default_repo "openai/clip-vit-base-patch32"
    @default_template "a photo of {label}"

    @doc """
    Classifies an image against a list of candidate labels.

    Wraps each label in the prompt template (default
    `"a photo of {label}"`), tokenises it, and asks CLIP which label
    best matches the image. Returns all labels scored and sorted
    descending.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `labels` is a non-empty list of label strings to classify
      against, e.g. `["a dog", "a cat", "a sports car"]`.

    * `options` is a keyword list of options.

    ### Options

    * `:repo` is the HuggingFace repository for the CLIP model. The
      default is `"#{@default_repo}"`.

    * `:template` is a prompt template — a string with `{label}` as a
      placeholder. The default is `"#{@default_template}"`. Pass
      `nil` to use labels verbatim.

    ### Returns

    * A list of `%{label: String.t(), score: float()}` maps sorted
      by descending `:score`. Scores sum to `1.0`.

    ### Examples

        iex> _puppy = Image.open!("./test/support/images/puppy.webp")
        iex> # Image.ZeroShot.classify(puppy, ["a dog", "a cat"])

    """
    @dialyzer {:nowarn_function, {:classify, 2}}
    @dialyzer {:nowarn_function, {:classify, 3}}

    @spec classify(Vimage.t(), [String.t(), ...], Keyword.t()) ::
            [%{label: String.t(), score: float()}]
    def classify(%Vimage{} = image, labels, options \\ [])
        when is_list(labels) and labels != [] do
      probabilities = predict_probabilities(image, labels, options)

      labels
      |> Enum.zip(probabilities)
      |> Enum.map(fn {label, score} -> %{label: label, score: score} end)
      |> Enum.sort_by(& &1.score, :desc)
    end

    @doc """
    Returns the single highest-scoring label for an image.

    Convenience wrapper over `classify/3` for the common "just tell
    me which one it is" case.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `labels` is a non-empty list of label strings.

    * `options` is a keyword list of options. Same as `classify/3`.

    ### Returns

    * The label string with the highest score.

    ### Examples

        iex> _puppy = Image.open!("./test/support/images/puppy.webp")
        iex> # Image.ZeroShot.label(puppy, ["a dog", "a cat"])
        iex> # => "a dog"

    """
    @dialyzer {:nowarn_function, {:label, 2}}
    @dialyzer {:nowarn_function, {:label, 3}}

    @spec label(Vimage.t(), [String.t(), ...], Keyword.t()) :: String.t()
    def label(%Vimage{} = image, labels, options \\ [])
        when is_list(labels) and labels != [] do
      [%{label: label} | _rest] = classify(image, labels, options)
      label
    end

    @doc """
    Computes cosine similarity between two images in CLIP's
    embedding space.

    Returns a value in `[-1.0, 1.0]` where higher means more
    visually similar. CLIP's image embeddings capture semantic
    content, so two images of different dogs typically score
    higher than a dog and a car, even if pixel-level differences
    are large.

    ### Arguments

    * `image1` and `image2` are any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:repo` is the HuggingFace repository for the CLIP model. The
      default is `"#{@default_repo}"`.

    ### Returns

    * A float in `[-1.0, 1.0]`.

    ### Examples

        iex> a = Image.open!("./test/support/images/puppy.webp")
        iex> b = Image.open!("./test/support/images/cat.png")
        iex> # Image.ZeroShot.similarity(a, b)

    """
    @dialyzer {:nowarn_function, {:similarity, 2}}
    @dialyzer {:nowarn_function, {:similarity, 3}}

    @spec similarity(Vimage.t(), Vimage.t(), Keyword.t()) :: float()
    def similarity(%Vimage{} = image1, %Vimage{} = image2, options \\ []) do
      {_model_info, featurizer, _tokenizer, _predict_fn, image_embed_fn, _text_embed_fn} =
        load_clip(options)

      e1 = image_embed_fn.(image1, featurizer)
      e2 = image_embed_fn.(image2, featurizer)

      Nx.dot(l2_normalize(e1), l2_normalize(e2)) |> Nx.to_number()
    end

    # --- Private --------------------------------------------------------

    defp predict_probabilities(image, labels, options) do
      template = Keyword.get(options, :template, @default_template)

      {model_info, featurizer, tokenizer, predict_fn, _image_embed_fn, _text_embed_fn} =
        load_clip(options)

      prompts = apply_template(labels, template)

      pixel_inputs = Bumblebee.apply_featurizer(featurizer, [image_to_stb(image)])
      text_inputs = Bumblebee.apply_tokenizer(tokenizer, prompts)

      inputs = Map.merge(text_inputs, pixel_inputs)

      outputs = predict_fn.(model_info.params, inputs)

      outputs.logits_per_image
      |> Nx.squeeze(axes: [0])
      |> Axon.Activations.softmax()
      |> Nx.to_list()
    end

    defp apply_template(labels, nil), do: labels

    defp apply_template(labels, template) do
      Enum.map(labels, &String.replace(template, "{label}", &1))
    end

    # Cache CLIP model_info, featurizer, tokenizer, and JIT-compiled
    # predict functions in :persistent_term keyed by repo. First call
    # downloads + compiles (slow); subsequent calls reuse everything.
    defp load_clip(options) do
      repo = Keyword.get(options, :repo, @default_repo)
      key = {__MODULE__, repo}

      case :persistent_term.get(key, nil) do
        nil ->
          {:ok, model_info} = Bumblebee.load_model({:hf, repo})
          {:ok, featurizer} = Bumblebee.load_featurizer({:hf, repo})
          {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repo})

          {_init_fn, predict_fn} = Axon.build(model_info.model)

          predict_fn = maybe_jit(predict_fn)

          image_embed_fn = fn image, feat ->
            inputs = Bumblebee.apply_featurizer(feat, [image_to_stb(image)])

            outputs =
              predict_fn.(
                model_info.params,
                Map.merge(inputs, %{
                  "input_ids" => Nx.tensor([[0]], type: :u32),
                  "attention_mask" => Nx.tensor([[1]], type: :u32)
                })
              )

            outputs.image_embedding |> Nx.squeeze(axes: [0])
          end

          text_embed_fn = fn texts, tok, feat ->
            inputs = Bumblebee.apply_tokenizer(tok, texts)

            zero_image =
              Nx.broadcast(
                Nx.tensor(0.0, type: :f32),
                {1, feat.size, feat.size, 3}
              )

            outputs =
              predict_fn.(
                model_info.params,
                Map.merge(inputs, %{"pixel_values" => zero_image})
              )

            outputs.text_embedding
          end

          cached = {model_info, featurizer, tokenizer, predict_fn, image_embed_fn, text_embed_fn}
          :persistent_term.put(key, cached)
          cached

        cached ->
          cached
      end
    end

    defp maybe_jit(fun) do
      if Code.ensure_loaded?(EXLA) do
        Nx.Defn.jit(fun, compiler: EXLA)
      else
        fun
      end
    end

    defp image_to_stb(%Vimage{} = image) do
      flat = image |> Image.flatten!() |> Image.to_colorspace!(:srgb)
      Image.to_nx!(flat, shape: :hwc, backend: Nx.BinaryBackend)
    end

    defp l2_normalize(tensor) do
      norm = tensor |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt() |> Nx.add(1.0e-12)
      Nx.divide(tensor, norm)
    end
  end
end
