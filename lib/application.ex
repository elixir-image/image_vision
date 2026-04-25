defmodule ImageVision.Application do
  @moduledoc false

  use Application
  require Logger

  @doc false
  def start(_type, _args) do
    Supervisor.start_link(
      children(Code.ensure_loaded?(Bumblebee)),
      strategy: :one_for_one,
      name: ImageVision.Supervisor
    )
  end

  # When Bumblebee is available, wire the classification service into
  # the supervision tree based on its `autostart:` config in
  # `:image_vision`. Segmentation and detection use Ortex, which loads
  # models lazily on first call — no supervised process needed.
  if ImageVision.bumblebee_configured?() do
    @services [
      {{Image.Classification, :classifier, []}, false},
      {{Image.Classification, :embedder, []}, false}
    ]

    defp children(true) do
      Enum.reduce(@services, [], fn {{module, function, args}, start?}, acc ->
        if autostart?(function, start?) do
          case apply(module, function, args) do
            {:error, reason} ->
              Logger.warning("Cannot autostart #{inspect(function)}. Error: #{inspect(reason)}")
              acc

            server ->
              [server | acc]
          end
        else
          acc
        end
      end)
    end
  end

  # When Bumblebee is not available, no children are started.
  defp children(_), do: []

  @doc false
  def autostart?(service, start?) do
    :image_vision
    |> Application.get_env(service, autostart: start?)
    |> Keyword.get(:autostart)
  end
end
