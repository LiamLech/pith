# Pith

A CLAP/VST3 audio effect plugin built with [NIH-plug](https://github.com/robbert-vdh/nih-plug).

Pith takes an FFT of the incoming signal and cuts frequency bins starting from
the lowest amplitude, morphing an instrument's timbre toward a pure sine wave.

> **Status:** beta. Expect rough edges.

## Parameters

| Parameter | Description |
|---|---|
| Cut Amount | Fraction of FFT bins to remove, smallest amplitude first (0-100%), shaped by Skew |
| Sine Amount | Blend of the synthesized sine wave (0 = cut-only signal, 100% = sine only) |
| Sine Mode | Behavior at Cut Amount = 100%: On keeps the single strongest bin, Off cuts everything (silence) |
| Skew Amount | Reshapes the Cut Amount curve for easier control at high values |
| Output Gain | Output gain, -24 dB to +24 dB |

## Building

Prebuilt binaries are available on the [Releases](../../releases) page. To build
from source, install [Rust](https://rustup.rs/) and the
[`cargo-nih-plug`](https://github.com/robbert-vdh/nih-plug/tree/master/cargo_nih_plug)
bundler:

```shell
cargo install --git https://github.com/robbert-vdh/nih-plug.git cargo-nih-plug
```

Then bundle the plugin:

```shell
cargo nih-plug bundle pith --release
```

The resulting `.clap` and `.vst3` files will be in `target/bundled/`.
