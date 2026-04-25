# Image Classification

`Image.Classification` answers "what's in this image?" and "how similar are these two images?".

## Getting labels

The simplest entry point is `labels/2`. It returns a list of human-readable labels for whatever is most prominent in the image:

```elixir
iex> puppy = Image.open!("puppy.jpg")
iex> Image.Classification.labels(puppy)
["Blenheim spaniel"]

iex> car = Image.open!("lamborghini.jpg")
iex> Image.Classification.labels(car)
["sports car", "sport car"]
```

Labels come from the model's training dataset (ImageNet-1k for the default ConvNeXt model — 1000 everyday categories). The default minimum confidence threshold is 0.5; adjust with `:min_score`:

```elixir
iex> Image.Classification.labels(puppy, min_score: 0.8)
["Blenheim spaniel"]

iex> Image.Classification.labels(puppy, min_score: 0.1)
["Blenheim spaniel", "cocker spaniel", "papillon"]
```

## Getting raw predictions with scores

`classify/2` returns the full prediction map including scores:

```elixir
iex> %{predictions: preds} = Image.Classification.classify(puppy)
iex> hd(preds)
%{label: "Blenheim spaniel", score: 0.9327}
```

## Computing embeddings

`embed/2` returns a 768-dimensional feature vector. Vectors for visually similar images will be close together in this space — useful for "find images like this one" or feeding into a custom classifier.

```elixir
iex> v1 = Image.Classification.embed(puppy)
iex> v2 = Image.Classification.embed(other_puppy)

# Cosine similarity: closer to 1.0 = more similar
iex> cos_sim = Nx.dot(v1, v2) / (Nx.norm(v1) * Nx.norm(v2)) |> Nx.to_number()
0.97
```

## Configuration

The classification serving is autostarted when the `:image_vision` application starts. To use a larger, more accurate model, set it in `config/runtime.exs`:

```elixir
config :image_vision, :classifier,
  model: {:hf, "facebook/convnext-large-224-22k-1k"},
  featurizer: {:hf, "facebook/convnext-large-224-22k-1k"}
```

The embedder is configured the same way, independently:

```elixir
config :image_vision, :embedder,
  model: {:hf, "facebook/dinov2-large"},
  featurizer: {:hf, "facebook/dinov2-large"}
```

Any image classification or image embedding model that Bumblebee can load works as a drop-in replacement — anything from the [HuggingFace image-classification](https://huggingface.co/models?pipeline_tag=image-classification) or [feature-extraction](https://huggingface.co/models?pipeline_tag=feature-extraction) catalogue with a corresponding featurizer. Larger models trade speed and memory for accuracy. The label set will be whatever the chosen model was trained on — for example, `convnext-large-224-22k-1k` returns the same 1000 ImageNet labels as the default, while a model fine-tuned on a different dataset returns that dataset's labels.

To manage the serving yourself (e.g. in an umbrella app):

```elixir
# config/runtime.exs
config :image_vision, :classifier, autostart: false

# application.ex
children = [Image.Classification.classifier()]
```

Configuration changes take effect at application start. After editing `config/runtime.exs`, restart the application; the new model is downloaded on first call and cached.

To pre-download a configured model so first-call latency is eliminated:

```bash
mix image_vision.download_models --classify
```

## Dependencies

Classification requires `:bumblebee`, `:nx`, and an Nx backend such as `:exla`. Add to `mix.exs`:

```elixir
{:bumblebee, "~> 0.6"},
{:exla, "~> 0.9"},
```
