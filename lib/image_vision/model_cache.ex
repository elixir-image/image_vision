defmodule ImageVision.ModelCache do
  @moduledoc """
  On-disk cache for ONNX model weights downloaded from
  [HuggingFace](https://huggingface.co).

  `Image.Segmentation` and `Image.Detection` use this cache to fetch
  their default models on first call. Files are downloaded once and
  reused for the lifetime of the cache directory.

  ## Cache directory

  The cache root is resolved in this order:

  * `Application.get_env(:image_vision, :cache_dir)` if set.

  * Otherwise, `:filename.basedir(:user_cache, "image_vision")` —
    on macOS this is `~/Library/Caches/image_vision`, on Linux it's
    `$XDG_CACHE_HOME/image_vision` (typically `~/.cache/image_vision`).

  Configure explicitly in `config/runtime.exs`:

      config :image_vision, :cache_dir, "/var/lib/image_vision/models"

  ## Layout

  Cached files are stored under `{cache_root}/{repo}/{filename}`,
  preserving the HuggingFace repository hierarchy. For example, the
  SAM 2.1 Tiny encoder lives at:

      {cache_root}/facebook/sam2.1-hiera-tiny/sam2.1_hiera_tiny.encoder.onnx

  ## Offline use

  Pre-populate the cache with the expected layout to operate fully
  offline. `cached?/2` reports whether a given file is present without
  triggering a download.

  """

  require Logger

  @hf_base_url "https://huggingface.co"

  @doc """
  Returns the local path for a HuggingFace model file, downloading it
  on cache miss.

  ### Arguments

  * `repo` is the HuggingFace repository, e.g. `"facebook/sam2.1-hiera-tiny"`.

  * `filename` is the path within the repository, e.g.
    `"sam2.1_hiera_tiny.encoder.onnx"`. Sub-directories are supported
    (`"onnx/encoder.onnx"`).

  ### Returns

  * The absolute local path to the file as a binary.

  Raises if the download fails.

  ### Examples

      iex> path = ImageVision.ModelCache.fetch!("facebook/sam2.1-hiera-tiny", "config.json")
      iex> File.exists?(path)
      true

  """
  @spec fetch!(String.t(), String.t()) :: Path.t()
  def fetch!(repo, filename) when is_binary(repo) and is_binary(filename) do
    target = path(repo, filename)

    if File.exists?(target) do
      target
    else
      download!(repo, filename, target)
      target
    end
  end

  @doc """
  Returns the local path that a given HuggingFace file would be cached
  at, without checking whether it exists or downloading it.

  ### Arguments

  * `repo` is the HuggingFace repository.

  * `filename` is the path within the repository.

  ### Returns

  * The absolute local path as a binary.

  """
  @spec path(String.t(), String.t()) :: Path.t()
  def path(repo, filename) when is_binary(repo) and is_binary(filename) do
    Path.join([cache_root(), repo, filename])
  end

  @doc """
  Returns true if the given HuggingFace file is already present in the
  local cache.

  ### Arguments

  * `repo` is the HuggingFace repository.

  * `filename` is the path within the repository.

  ### Returns

  * A boolean.

  """
  @spec cached?(String.t(), String.t()) :: boolean()
  def cached?(repo, filename) when is_binary(repo) and is_binary(filename) do
    repo |> path(filename) |> File.exists?()
  end

  @doc """
  Returns the absolute path to the cache root directory, creating it
  if it does not yet exist.

  ### Returns

  * The absolute path as a binary.

  """
  @spec cache_root() :: Path.t()
  def cache_root do
    root =
      case Application.get_env(:image_vision, :cache_dir) do
        nil ->
          :user_cache
          |> :filename.basedir(~c"image_vision")
          |> List.to_string()

        configured when is_binary(configured) ->
          configured
      end

    File.mkdir_p!(root)
    root
  end

  # Streams a file from HuggingFace to a temporary path, then atomically
  # renames it to `target`. Two concurrent processes may both download
  # the same file on first call — the final rename remains correct.

  defp download!(repo, filename, target) do
    url = "#{@hf_base_url}/#{repo}/resolve/main/#{filename}"
    parent = Path.dirname(target)
    File.mkdir_p!(parent)

    tmp = "#{target}.#{:erlang.unique_integer([:positive])}.tmp"
    Logger.info("[image_vision] downloading #{repo}/#{filename}")

    try do
      response =
        Req.get!(url,
          into: File.stream!(tmp),
          redirect: true,
          receive_timeout: 600_000
        )

      if response.status not in 200..299 do
        raise "HuggingFace download failed with status #{response.status} for #{url}"
      end

      File.rename!(tmp, target)
    rescue
      error ->
        File.rm(tmp)
        reraise error, __STACKTRACE__
    end

    Logger.info("[image_vision] cached #{repo}/#{filename} at #{target}")
    :ok
  end
end
