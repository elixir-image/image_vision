if ImageVision.bumblebee_configured?() do
  defmodule Image.Captioning do
    @moduledoc """
    Image captioning — generates a natural-language description of an
    image.

    Pass a `t:Vix.Vips.Image.t/0` to `caption/2` and get back a string
    like `"a small dog sitting on a wooden floor"` or `"a man riding
    a horse with a bird of prey"`.

    ## Quick start

        # The captioner serving is heavyweight and not autostarted by
        # default. Either set `autostart: true` in config (see below)
        # or add the child spec to your own supervision tree:
        #
        #     children = [Image.Captioning.captioner()]

        iex> _puppy = Image.open!("./test/support/images/puppy.webp")
        iex> # Image.Captioning.caption(puppy)
        iex> # => "a brown and white puppy sitting on a white surface"

    ## Default model

    [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
    — BSD-3-Clause licensed, ~990 MB. The base BLIP variant fine-tuned
    for image captioning. Solid baseline quality across general subject
    matter.

    Note that this is by far the heaviest of the library's default
    models — the first call (or first app boot with `autostart: true`)
    blocks on a ~990 MB download from HuggingFace.

    ## Configuration

    Configure in `config/runtime.exs`:

        config :image_vision, :captioner,
          model: {:hf, "Salesforce/blip-image-captioning-base"},
          featurizer: {:hf, "Salesforce/blip-image-captioning-base"},
          tokenizer: {:hf, "Salesforce/blip-image-captioning-base"},
          generation_config: {:hf, "Salesforce/blip-image-captioning-base"},
          model_options: [],
          featurizer_options: [],
          tokenizer_options: [],
          generation_config_options: [],
          batch_size: 1,
          name: Image.Captioning.Server,
          autostart: false

    To use the larger and higher-quality variant:

        config :image_vision, :captioner,
          model: {:hf, "Salesforce/blip-image-captioning-large"},
          featurizer: {:hf, "Salesforce/blip-image-captioning-large"},
          tokenizer: {:hf, "Salesforce/blip-image-captioning-large"},
          generation_config: {:hf, "Salesforce/blip-image-captioning-large"}

    ## Servings and supervision

    BLIP is a multi-module model (vision encoder, text decoder,
    cross-attention) and a load takes several seconds. The captioning
    entry point runs against a named serving process so the model
    loads once and is reused.

    The serving is not autostarted by default — most apps either don't
    need captioning at all or want explicit control over when the
    download happens. To run it in your own supervision tree:

        # application.ex
        def start(_type, _args) do
          children = [Image.Captioning.captioner()]
          Supervisor.start_link(children, strategy: :one_for_one)
        end

    Or set `autostart: true` to have `ImageVision.Supervisor` start it
    when the `:image_vision` application starts.

    ## Optional dependency

    This module is only available when [Bumblebee](https://hex.pm/packages/bumblebee),
    [Nx](https://hex.pm/packages/nx), and an Nx compiler such as
    [EXLA](https://hex.pm/packages/exla) are configured in your
    application's `mix.exs`.

    """

    alias Vix.Vips.Image, as: Vimage

    @blip_repo "Salesforce/blip-image-captioning-base"

    @default_captioner [
      model: {:hf, @blip_repo},
      featurizer: {:hf, @blip_repo},
      tokenizer: {:hf, @blip_repo},
      generation_config: {:hf, @blip_repo},
      model_options: [],
      featurizer_options: [],
      tokenizer_options: [],
      generation_config_options: [],
      name: Image.Captioning.Server,
      batch_size: 1,
      autostart: false
    ]

    @default_captioner_name @default_captioner[:name]

    @doc """
    Returns a child spec suitable for starting an image captioning
    process as part of a supervision tree.

    ### Arguments

    * `configuration` is a keyword list merged over the default
      configuration.

    ### Options

    * `:model` is any BLIP-family image captioning model supported by
      Bumblebee. The default is `{:hf, "#{@blip_repo}"}`.

    * `:featurizer` is the BLIP featurizer. The default is
      `{:hf, "#{@blip_repo}"}`.

    * `:tokenizer` is the BLIP tokenizer. The default is
      `{:hf, "#{@blip_repo}"}`.

    * `:generation_config` is a Bumblebee generation config repo. The
      default is `{:hf, "#{@blip_repo}"}`.

    * `:model_options`, `:featurizer_options`, `:tokenizer_options`,
      and `:generation_config_options` are keyword lists passed to
      the corresponding `Bumblebee.load_*` functions. Defaults are `[]`.

    * `:name` is the name of the serving process. The default is
      `Image.Captioning.Server`.

    * `:batch_size` is the maximum batch size. The default is `1`.

    ### Returns

    * A child spec tuple suitable for `Supervisor.start_link/2`, or

    * `{:error, reason}` if the model could not be loaded.

    """
    @spec captioner(configuration :: Keyword.t()) ::
            {Nx.Serving, Keyword.t()} | {:error, Image.error()}
    def captioner(configuration \\ Application.get_env(:image_vision, :captioner, [])) do
      configuration = Keyword.merge(@default_captioner, configuration)

      with {:ok, model_info} <-
             Bumblebee.load_model(
               configuration[:model],
               configuration[:model_options]
             ),
           {:ok, featurizer} <-
             Bumblebee.load_featurizer(
               configuration[:featurizer],
               configuration[:featurizer_options]
             ),
           {:ok, tokenizer} <-
             Bumblebee.load_tokenizer(
               configuration[:tokenizer],
               configuration[:tokenizer_options]
             ),
           {:ok, generation_config} <-
             Bumblebee.load_generation_config(
               configuration[:generation_config],
               configuration[:generation_config_options]
             ) do
        serving =
          Bumblebee.Vision.image_to_text(
            model_info,
            featurizer,
            tokenizer,
            generation_config,
            compile: [batch_size: configuration[:batch_size]],
            defn_options: defn_options()
          )

        {Nx.Serving, serving: serving, name: configuration[:name], batch_timeout: 100}
      end
    end

    defp defn_options do
      if Code.ensure_loaded?(EXLA) do
        [compiler: EXLA]
      else
        []
      end
    end

    @doc """
    Generates a natural-language caption for an image.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:backend` is any valid Nx backend used for the image-to-tensor
      conversion. The default is `Nx.default_backend/0`.

    * `:server` is the name of the captioning serving process. The
      default is `Image.Captioning.Server`.

    ### Returns

    * The caption as a `t:String.t/0`, or

    * `{:error, reason}` if the input could not be processed.

    ### Examples

        iex> _puppy = Image.open!("./test/support/images/puppy.webp")
        iex> # Image.Captioning.caption(puppy)
        iex> # => "a small dog sitting on a wooden surface"

    """
    @dialyzer {:nowarn_function, {:caption, 1}}
    @dialyzer {:nowarn_function, {:caption, 2}}

    @spec caption(image :: Vimage.t(), options :: Keyword.t()) ::
            String.t() | {:error, Image.error()}
    def caption(%Vimage{} = image, options \\ []) do
      backend = Keyword.get(options, :backend, Nx.default_backend())
      server = Keyword.get(options, :server, @default_captioner_name)

      with {:ok, tensor} <- to_caption_tensor(image, backend),
           %{results: [%{text: text} | _rest]} <- Nx.Serving.batched_run(server, tensor) do
        text
      end
    end

    defp to_caption_tensor(%Vimage{} = image, backend) do
      with {:ok, flat} <- Image.flatten(image),
           {:ok, srgb} <- Image.to_colorspace(flat, :srgb) do
        Image.to_nx(srgb, shape: :hwc, backend: backend)
      end
    end
  end
end
