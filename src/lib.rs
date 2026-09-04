use nih_plug::prelude::*;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

const WINDOW_SIZE: usize = 8192;
const HOP_SIZE: usize = 2048;
const NUM_BINS: usize = WINDOW_SIZE / 2 + 1;

struct ChannelState {
    input_ring: Vec<f32>,
    ring_write_pos: usize,
    samples_until_process: usize,
    output_buffer: Vec<f32>,
    output_read_pos: usize,
    complex_fft_buffer: Vec<Complex32>,
    fft_input: Vec<f32>,
    ifft_output: Vec<f32>,
    amplitudes: Vec<(usize, f32)>,
    sine_phase: f32,
    current_freq: f32,
    frame_rms: f32,
    input_rms: f32,
}

impl ChannelState {
    fn new(r2c_plan: &dyn RealToComplex<f32>) -> Self {
        Self {
            input_ring: vec![0.0f32; WINDOW_SIZE],
            ring_write_pos: 0,
            samples_until_process: HOP_SIZE,
            output_buffer: vec![0.0f32; WINDOW_SIZE * 2],
            output_read_pos: 0,
            complex_fft_buffer: r2c_plan.make_output_vec(),
            fft_input: vec![0.0f32; WINDOW_SIZE],
            ifft_output: vec![0.0f32; WINDOW_SIZE],
            amplitudes: vec![(0, 0.0f32); NUM_BINS],
            sine_phase: 0.0,
            current_freq: 440.0,
            frame_rms: 0.0,
            input_rms: 0.0,
        }
    }

    fn reset(&mut self) {
        self.input_ring.fill(0.0);
        self.ring_write_pos = 0;
        self.samples_until_process = HOP_SIZE;
        self.output_buffer.fill(0.0);
        self.output_read_pos = 0;
        self.sine_phase = 0.0;
        self.current_freq = 440.0;
        self.frame_rms = 0.0;
        self.input_rms = 0.0;
    }

    fn estimate_pitch(&self, sample_rate: f32) -> Option<f32> {
        let amplitudes: Vec<f32> = self.complex_fft_buffer
            .iter()
            .map(|bin| bin.norm())
            .collect();

        let min_bin = (50.0 * WINDOW_SIZE as f32 / sample_rate) as usize;
        let max_bin = (2000.0 * WINDOW_SIZE as f32 / sample_rate) as usize;
        let max_bin = max_bin.min(NUM_BINS - 2);

        if min_bin >= max_bin {
            return None;
        }

        let dominant_bin = (min_bin..max_bin)
            .max_by(|&a, &b| {
                amplitudes[a].partial_cmp(&amplitudes[b]).unwrap()
            })?;

        let max_amp = amplitudes[dominant_bin];
        if max_amp < 1e-4 {
            return None;
        }

        if dominant_bin > 0 && dominant_bin < NUM_BINS - 1 {
            let alpha = amplitudes[dominant_bin - 1];
            let beta = amplitudes[dominant_bin];
            let gamma = amplitudes[dominant_bin + 1];
            let denom = alpha - 2.0 * beta + gamma;
            let correction = if denom.abs() > 1e-10 {
                0.5 * (alpha - gamma) / denom
            } else {
                0.0
            };
            let precise_bin = dominant_bin as f32 + correction;
            Some(precise_bin * sample_rate / WINDOW_SIZE as f32)
        } else {
            Some(dominant_bin as f32 * sample_rate / WINDOW_SIZE as f32)
        }
    }
}

struct Pith {
    params: Arc<PithParams>,
    r2c_plan: Arc<dyn RealToComplex<f32>>,
    c2r_plan: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    channels: [ChannelState; 2],
    sample_rate: f32,
}

#[derive(Params)]
struct PithParams {
    #[id = "cut_amount"]
    pub cut_amount: FloatParam,
    #[id = "sine_amount"]
    pub sine_amount: FloatParam,
    #[id = "sine_mode"]
    pub sine_mode: BoolParam,
    #[id = "skew_amount"]
    pub skew_amount: FloatParam,
    #[id = "output_gain"]
    pub output_gain: FloatParam,
}

impl Default for Pith {
    fn default() -> Self {
        let mut planner = RealFftPlanner::new();
        let r2c_plan = planner.plan_fft_forward(WINDOW_SIZE);
        let c2r_plan = planner.plan_fft_inverse(WINDOW_SIZE);

        let window: Vec<f32> = (0..WINDOW_SIZE)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32
                        / WINDOW_SIZE as f32)
                        .cos())
            })
            .collect();

        let channels = [
            ChannelState::new(r2c_plan.as_ref()),
            ChannelState::new(r2c_plan.as_ref()),
        ];

        Self {
            params: Arc::new(PithParams::default()),
            r2c_plan,
            c2r_plan,
            window,
            channels,
            sample_rate: 48000.0,
        }
    }
}

impl Default for PithParams {
    fn default() -> Self {
        Self {
            cut_amount: FloatParam::new(
                "Cut Amount",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(1))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            sine_amount: FloatParam::new(
                "Sine Amount",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(1))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            sine_mode: BoolParam::new("Sine Mode", false),
            skew_amount: FloatParam::new(
                "Skew Amount",
                1.0,
                FloatRange::Linear { min: 0.0, max: 5.0 },
            )
            .with_step_size(0.1),
            output_gain: FloatParam::new(
                "Output Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-24.0),
                    max: util::db_to_gain(24.0),
                    factor: FloatRange::gain_skew_factor(-24.0, 24.0),
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
        }
    }
}

impl Plugin for Pith {
    const NAME: &'static str = "Pith-008";
    const VENDOR: &'static str = "";
    const URL: &'static str = "";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;
        context.set_latency_samples(WINDOW_SIZE as u32);
        true
    }

    fn reset(&mut self) {
        for ch in self.channels.iter_mut() {
            ch.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let cut_amount_raw = self.params.cut_amount.smoothed.next();
        let sine_amount = self.params.sine_amount.value();
        let skew = self.params.skew_amount.value();
        let output_gain = self.params.output_gain.value();
        let sine_mode = self.params.sine_mode.value();
        let sample_rate = self.sample_rate;

        let cut_amount = cut_amount_raw.powf(1.0 / (skew + 1.0));

        let num_samples = buffer.samples();
        let r2c_plan = &self.r2c_plan;
        let c2r_plan = &self.c2r_plan;
        let window = &self.window;

        for ch_idx in 0..2 {
            let ch = &mut self.channels[ch_idx];
            let channel_samples = &mut buffer.as_slice()[ch_idx];

            for sample_idx in 0..num_samples {
                let input_sample = channel_samples[sample_idx];

                ch.input_ring[ch.ring_write_pos] = input_sample;
                ch.ring_write_pos = (ch.ring_write_pos + 1) % WINDOW_SIZE;

                ch.samples_until_process -= 1;

                if ch.samples_until_process == 0 {
                    ch.samples_until_process = HOP_SIZE;

                    for i in 0..WINDOW_SIZE {
                        let ring_pos = (ch.ring_write_pos + i) % WINDOW_SIZE;
                        ch.fft_input[i] = ch.input_ring[ring_pos] * window[i];
                    }

                    let input_rms_frame = (ch.fft_input.iter()
                        .map(|s| s * s)
                        .sum::<f32>()
                        / WINDOW_SIZE as f32)
                        .sqrt();
                    ch.input_rms = ch.input_rms * 0.8 + input_rms_frame * 0.2;

                    r2c_plan
                        .process_with_scratch(
                            &mut ch.fft_input,
                            &mut ch.complex_fft_buffer,
                            &mut [],
                        )
                        .unwrap();

                    // ピッチ推定
                    if sine_amount > 0.0 {
                        if let Some(freq) = ch.estimate_pitch(sample_rate) {
                            ch.current_freq = ch.current_freq * 0.7
                                + freq * 0.3;
                        }
                    }

                    // 各ビンの振幅を計算してソート
                    for i in 0..NUM_BINS {
                        ch.amplitudes[i] =
                            (i, ch.complex_fft_buffer[i].norm());
                    }
                    ch.amplitudes
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    // カット処理
                    let noise_floor = 1e-6_f32;
                    let significant_count = ch
                        .amplitudes
                        .iter()
                        .filter(|(_, amp)| *amp > noise_floor)
                        .count();

                    let is_full_cut = cut_amount_raw >= 0.999;

                    let cut_count = if is_full_cut {
                        if sine_mode {
                            significant_count.saturating_sub(1)
                        } else {
                            NUM_BINS
                        }
                    } else {
                        (significant_count as f32 * cut_amount) as usize
                    };

                    for i in 0..cut_count.min(NUM_BINS) {
                        let bin_idx = ch.amplitudes[i].0;
                        ch.complex_fft_buffer[bin_idx] =
                            Complex32::new(0.0, 0.0);
                    }

                    // IFFT
                    c2r_plan
                        .process_with_scratch(
                            &mut ch.complex_fft_buffer,
                            &mut ch.ifft_output,
                            &mut [],
                        )
                        .unwrap();

                    // フレームのRMSを計算
                    let rms = (ch.ifft_output.iter()
                        .map(|s| s * s)
                        .sum::<f32>()
                        / WINDOW_SIZE as f32)
                        .sqrt();
                    ch.frame_rms = ch.frame_rms * 0.7 + rms * 0.3;

                    // オーバーラップアド
                    let write_pos = ch.output_read_pos;
                    for i in 0..WINDOW_SIZE {
                        let pos = (write_pos + i) % (WINDOW_SIZE * 2);
                        ch.output_buffer[pos] +=
                            ch.ifft_output[i] * window[i] / WINDOW_SIZE as f32;
                    }
                }

                // Mode A出力
                let mode_a_sample = ch.output_buffer[ch.output_read_pos];
                ch.output_buffer[ch.output_read_pos] = 0.0;
                ch.output_read_pos =
                    (ch.output_read_pos + 1) % (WINDOW_SIZE * 2);

                // Mode B：サイン波合成
                let sine_sample = ch.sine_phase.sin();
                ch.sine_phase += 2.0 * std::f32::consts::PI
                    * ch.current_freq / sample_rate;
                if ch.sine_phase > 2.0 * std::f32::consts::PI {
                    ch.sine_phase -= 2.0 * std::f32::consts::PI;
                }

                // Mode AとMode Bを混合
                let output_sample = if sine_amount > 0.0 {
                    let sine_scaled = sine_sample * ch.input_rms;
                    let a_gain = 1.0 - sine_amount;
                    let b_gain = sine_amount;
                    mode_a_sample * a_gain + sine_scaled * b_gain
                } else {
                    mode_a_sample
                };

                channel_samples[sample_idx] = output_sample * output_gain;
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for Pith {
    const CLAP_ID: &'static str = "com.pith.pith-008";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Pith-008");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for Pith {
    const VST3_CLASS_ID: [u8; 16] = *b"PithPlugin008AAA";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx];
}

nih_export_clap!(Pith);
nih_export_vst3!(Pith);