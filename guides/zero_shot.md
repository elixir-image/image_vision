# Zero-Shot Classification

`Image.ZeroShot` lets you classify an image against arbitrary labels you provide at call time, without training or fine-tuning anything. Where `Image.Classification` is constrained to whatever 1000 ImageNet labels its model was trained on, zero-shot says "here are five categories I care about right now — which fits best?".

This is enormously useful when your label space:

- Doesn't exist in standard datasets (custom product categories, brand-specific taxonomies, ad-hoc tagging)
- Changes over time (new categories appear without retraining)
- Is unknown until query time (interactive search, user-driven filtering)

Powered by [CLIP](https://openai.com/research/clip), a contrastive vision-language model that learned a shared embedding space for images and text from 400 million image-caption pairs.

## Classifying

```elixir
iex> photo = Image.open!("portrait.jpg")
iex> Image.ZeroShot.classify(photo, [
...>   "a person on a horse",
...>   "a person walking a dog",
...>   "a parked car",
...>   "an empty street"
...> ])
[
  %{label: "a person on a horse", score: 0.94},
  %{label: "a person walking a dog", score: 0.04},
  %{label: "an empty street", score: 0.01},
  %{label: "a parked car", score: 0.01}
]
```

Scores sum to `1.0` (softmax over the candidate set). Results are sorted by descending score.

## Just the best label

When you only want the winner:

```elixir
iex> Image.ZeroShot.label(photo, ["dog", "cat", "horse"])
"horse"
```

## Image-to-image similarity

CLIP's image embeddings live in the same space as its text embeddings, so two images can also be compared directly:

```elixir
iex> a = Image.open!("dog1.jpg")
iex> b = Image.open!("dog2.jpg")
iex> c = Image.open!("car.jpg")
iex>
iex> Image.ZeroShot.similarity(a, b)
0.82
iex> Image.ZeroShot.similarity(a, c)
0.41
```

Useful for "find similar images" without standing up a vector database. For larger collections, compute embeddings once and cache them.

## Prompt templates matter

CLIP was trained on natural-language captions, so it understands sentences much better than bare nouns. Wrapping each label in a simple template reliably improves accuracy. The default is `"a photo of {label}"` — every label is wrapped before tokenisation.

You can override:

```elixir
# Photography-domain template
iex> Image.ZeroShot.classify(image, ["sunset", "rain", "snow"],
...>   template: "a high-quality photograph of {label}")

# Document-domain template
iex> Image.ZeroShot.classify(scan, ["invoice", "receipt", "letter"],
...>   template: "a scanned {label}")

# Disable templating if your labels are already full sentences
iex> Image.ZeroShot.classify(image,
...>   ["a black cat sitting on a chair", "an empty room with a chair"],
...>   template: nil)
```

A general rule: if your labels are nouns, leave the default template. If they're already descriptive sentences, use `template: nil`. If they're domain-specific (medical imagery, document scans, product photography), a domain-tailored template can help.

## Choosing a model

The default is `openai/clip-vit-base-patch32` — MIT licensed, ~600 MB, broad training coverage, well-validated. For higher quality at larger size:

```elixir
iex> Image.ZeroShot.classify(image, labels, repo: "openai/clip-vit-large-patch14")
```

(About ~1.7 GB — roughly 3× the default.)

## Pre-downloading

```bash
mix image_vision.download_models --zero-shot
```

## Default model

[OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) — MIT licensed, ~600 MB. The original CLIP, the most well-validated and broadly applicable variant. Contains both a vision encoder (ViT-B/32) and a text encoder (transformer) that produce vectors in a shared 512-dim space.

## Dependencies

Zero-shot classification requires `:bumblebee`, `:nx`, and an Nx backend such as `:exla`. Add to `mix.exs`:

```elixir
{:bumblebee, "~> 0.6"},
{:nx, "~> 0.10"},
{:exla, "~> 0.10"}
```
