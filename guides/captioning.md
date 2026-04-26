# Image Captioning

`Image.Captioning` answers "describe this image in plain English". Useful for accessibility (alt text), image search and indexing (caption as a search field), assistive tooling, content moderation pipelines that want a quick natural-language summary, and any product surface where you'd otherwise need a human to write descriptions for thousands of images.

## Quick start

The captioner is a heavyweight Bumblebee serving — it loads ~990 MB of weights and JIT-compiles a transformer decoder. It is **not** autostarted by default. Before calling `caption/2`, either configure autostart:

```elixir
# config/runtime.exs
config :image_vision, :captioner, autostart: true
```

…or add the child spec to your own supervision tree:

```elixir
# application.ex
children = [Image.Captioning.captioner()]
Supervisor.start_link(children, strategy: :one_for_one)
```

Then:

```elixir
iex> photo = Image.open!("photo.jpg")
iex> Image.Captioning.caption(photo)
"a man riding a horse with a bird of prey on his arm"
```

## Choosing a model

The default is BLIP base (`Salesforce/blip-image-captioning-base`, BSD-3-Clause, ~990 MB) — a solid baseline that produces concise, generally accurate captions.

For higher-quality captions at ~3× the size:

```elixir
config :image_vision, :captioner,
  model: {:hf, "Salesforce/blip-image-captioning-large"},
  featurizer: {:hf, "Salesforce/blip-image-captioning-large"},
  tokenizer: {:hf, "Salesforce/blip-image-captioning-large"},
  generation_config: {:hf, "Salesforce/blip-image-captioning-large"}
```

Avoid BLIP-2 (`Salesforce/blip2-*`) — its OPT decoder has non-commercial licensing.

## Tuning generation

The default generation config is whatever the model ships with — usually around 20 tokens, greedy decoding. To tune (more tokens, beam search, sampling), set the relevant fields on the generation config via Bumblebee's loaded config and configure them at serving construction time. For most use cases the defaults are fine.

## Pre-downloading

The default BLIP weights are ~990 MB — by far the heaviest of the library's defaults. Pre-download to avoid blocking on first call:

```bash
mix image_vision.download_models --caption
```

## Why captioning is heavy

A vision classifier runs the encoder once per image and is done. A captioner runs the encoder once, then runs the text decoder *autoregressively* — one forward pass per generated token. A 20-word caption is ~20 forward passes through the decoder. That's why generation is slower than classification, and why we autostart the serving so the model only loads once across the lifetime of the application.

## Default model

[BLIP base captioning](https://huggingface.co/Salesforce/blip-image-captioning-base) — Salesforce, BSD-3-Clause licensed, ~990 MB. Trained on 129M image-text pairs. The base variant trades some descriptive richness for size and speed compared to the large variant.

## Dependencies

Captioning requires `:bumblebee`, `:nx`, and an Nx backend such as `:exla`. Add to `mix.exs`:

```elixir
{:bumblebee, "~> 0.6"},
{:nx, "~> 0.10"},
{:exla, "~> 0.10"}
```
