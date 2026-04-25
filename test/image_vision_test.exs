defmodule ImageVisionTest do
  use ExUnit.Case

  doctest ImageVision

  test "bumblebee_configured? returns a boolean" do
    assert is_boolean(ImageVision.bumblebee_configured?())
  end

  test "ortex_configured? returns a boolean" do
    assert is_boolean(ImageVision.ortex_configured?())
  end
end
