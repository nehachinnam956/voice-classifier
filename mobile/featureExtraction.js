/**
 * featureExtraction.js — JS port of src/data_prep.py::extract_mel_spectrogram.
 *
 * Self-contained: implements its own radix-2 FFT and mel filterbank rather
 * than pulling in a DSP dependency, so there's nothing else to install or
 * version-match. Matches librosa's defaults used in training:
 *   - window: Hann
 *   - center=True, pad_mode='constant' (zero-pad by n_fft//2 each side)
 *   - mel scale: Slaney (htk=False), Slaney-normalized filter weights
 *   - power_to_db with ref=max
 *
 * VERIFIED: this exact algorithm was checked against a real librosa
 * reference array and matched within numerical tolerance for a synthetic
 * test clip (see the verification script referenced in the repo). Re-verify
 * with a real recorded word before shipping — synthetic sine waves don't
 * exercise every code path (e.g. the trim-silence step is not yet ported;
 * see TODO at the bottom).
 */

export const SAMPLE_RATE = 16000;
export const CLIP_SAMPLES = 16000; // 1.0s
export const N_MELS = 64;
export const N_FFT = 512;
export const HOP_LENGTH = 160;
const FMIN = 0.0;
const FMAX = SAMPLE_RATE / 2;

// ---------- FFT (iterative radix-2, in-place) ----------
function fft(real, imag) {
  const n = real.length;
  if (n <= 1) return;

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curWr = 1, curWi = 0;
      for (let j = 0; j < len / 2; j++) {
        const ur = real[i + j], ui = imag[i + j];
        const vr = real[i + j + len / 2] * curWr - imag[i + j + len / 2] * curWi;
        const vi = real[i + j + len / 2] * curWi + imag[i + j + len / 2] * curWr;
        real[i + j] = ur + vr;
        imag[i + j] = ui + vi;
        real[i + j + len / 2] = ur - vr;
        imag[i + j + len / 2] = ui - vi;
        const nextWr = curWr * wr - curWi * wi;
        curWi = curWr * wi + curWi * wr;
        curWr = nextWr;
      }
    }
  }
}

function hannWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  }
  return w;
}

/** Zero-pad n_fft//2 samples on each side (librosa center=True, pad_mode='constant'). */
function centerPad(samples, padAmount) {
  const out = new Float32Array(samples.length + 2 * padAmount);
  out.set(samples, padAmount);
  return out;
}

/** Power spectrogram: shape [n_frames][n_fft/2 + 1]. */
function stftPower(samples) {
  const padAmount = Math.floor(N_FFT / 2);
  const padded = centerPad(samples, padAmount);
  const window = hannWindow(N_FFT);
  const nFrames = 1 + Math.floor((padded.length - N_FFT) / HOP_LENGTH);
  const nBins = N_FFT / 2 + 1;
  const power = [];

  for (let f = 0; f < nFrames; f++) {
    const start = f * HOP_LENGTH;
    const real = new Float64Array(N_FFT);
    const imag = new Float64Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
      real[i] = padded[start + i] * window[i];
    }
    fft(real, imag);
    const frame = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) {
      frame[k] = real[k] * real[k] + imag[k] * imag[k];
    }
    power.push(frame);
  }
  return power;
}

// ---------- Slaney mel filterbank (matches librosa htk=False, norm='slaney') ----------
function hzToMel(hz) {
  const fMin = 0, fSp = 200 / 3;
  let mel = (hz - fMin) / fSp;
  const minLogHz = 1000.0, minLogMel = (minLogHz - fMin) / fSp, logstep = Math.log(6.4) / 27.0;
  if (hz >= minLogHz) {
    mel = minLogMel + Math.log(hz / minLogHz) / logstep;
  }
  return mel;
}

function melToHz(mel) {
  const fMin = 0, fSp = 200 / 3;
  let hz = fMin + fSp * mel;
  const minLogHz = 1000.0, minLogMel = (minLogHz - fMin) / fSp, logstep = Math.log(6.4) / 27.0;
  if (mel >= minLogMel) {
    hz = minLogHz * Math.exp(logstep * (mel - minLogMel));
  }
  return hz;
}

function buildMelFilterbank(sr, nFft, nMels, fmin, fmax) {
  const nBins = Math.floor(nFft / 2) + 1;
  const fftFreqs = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) fftFreqs[i] = (i * sr) / nFft;

  const melMin = hzToMel(fmin), melMax = hzToMel(fmax);
  const melPoints = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    melPoints[i] = melMin + ((melMax - melMin) * i) / (nMels + 1);
  }
  const hzPoints = Array.from(melPoints, melToHz);

  const weights = Array.from({ length: nMels }, () => new Float64Array(nBins));
  for (let m = 0; m < nMels; m++) {
    const fLeft = hzPoints[m], fCenter = hzPoints[m + 1], fRight = hzPoints[m + 2];
    for (let k = 0; k < nBins; k++) {
      const f = fftFreqs[k];
      let w = 0;
      if (f >= fLeft && f <= fCenter && fCenter !== fLeft) {
        w = (f - fLeft) / (fCenter - fLeft);
      } else if (f > fCenter && f <= fRight && fRight !== fCenter) {
        w = (fRight - f) / (fRight - fCenter);
      }
      weights[m][k] = w;
    }
    const enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m]);
    for (let k = 0; k < nBins; k++) weights[m][k] *= enorm;
  }
  return weights;
}

let _melFilterbank = null;
function getMelFilterbank() {
  if (!_melFilterbank) {
    _melFilterbank = buildMelFilterbank(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX);
  }
  return _melFilterbank;
}

/**
 * Full pipeline: fixed-length float samples (already trimmed/normalized/
 * padded to CLIP_SAMPLES, see TODO below) -> log-mel spectrogram, shape
 * [N_MELS][nFrames], matching data_prep.py::extract_mel_spectrogram.
 */
export function melSpectrogramFromSamples(samples) {
  const power = stftPower(samples);
  const filterbank = getMelFilterbank();
  const nFrames = power.length;

  const mel = Array.from({ length: N_MELS }, () => new Float64Array(nFrames));
  for (let m = 0; m < N_MELS; m++) {
    for (let f = 0; f < nFrames; f++) {
      let sum = 0;
      const fbRow = filterbank[m];
      const frame = power[f];
      for (let k = 0; k < fbRow.length; k++) sum += fbRow[k] * frame[k];
      mel[m][f] = sum;
    }
  }

  let maxVal = 1e-10;
  for (let m = 0; m < N_MELS; m++)
    for (let f = 0; f < nFrames; f++)
      if (mel[m][f] > maxVal) maxVal = mel[m][f];

  const logMel = Array.from({ length: N_MELS }, () => new Float32Array(nFrames));
  for (let m = 0; m < N_MELS; m++) {
    for (let f = 0; f < nFrames; f++) {
      const s = Math.max(mel[m][f], 1e-10);
      let db = 10 * Math.log10(s) - 10 * Math.log10(maxVal);
      db = Math.max(db, -80.0);
      logMel[m][f] = db;
    }
  }
  return logMel;
}

/**
 * @param {Float32Array} pcmSamples - decoded PCM samples, already resampled
 *   to 16kHz mono in [-1, 1] range.
 * @returns {Float32Array} flattened (N_MELS * nFrames) log-mel, ready to
 *   reshape to the model's [1, N_MELS, nFrames, 1] input.
 *
 * TODO before shipping: this function assumes `pcmSamples` is already
 * trim-silenced, peak-normalized, and fixed to CLIP_SAMPLES length — i.e.
 * it only ports extract_mel_spectrogram, not the full preprocess_audio
 * chain from data_prep.py. Port trim_silence/normalize_amplitude/fix_length
 * too (they're simpler than the FFT/mel code above) before wiring this
 * into App.js's stopRecordingAndClassify. Reasonable v1 shortcut: skip
 * silence trimming on-device (a fixed 1s window is usually enough) but do
 * implement peak-normalize and fix_length — those are ~10 lines each.
 */
/** Peak-normalize to [-1, 1]. Mirrors data_prep.normalize_amplitude. */
export function normalizeAmplitude(samples) {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i]));
  if (peak < 1e-6) return samples;
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) out[i] = samples[i] / peak;
  return out;
}

/**
 * Center-crop or symmetric zero-pad to CLIP_SAMPLES.
 * Mirrors data_prep.fix_length exactly (crop favors the center, not the
 * start, so a word starting slightly late isn't chopped off).
 */
export function fixLength(samples, target = CLIP_SAMPLES) {
  const n = samples.length;
  if (n === target) return samples;
  if (n < target) {
    const padTotal = target - n;
    const padLeft = Math.floor(padTotal / 2);
    const out = new Float32Array(target);
    out.set(samples, padLeft);
    return out;
  }
  const start = Math.floor((n - target) / 2);
  return samples.slice(start, start + target);
}

/**
 * Decode a base64-encoded 16-bit PCM WAV file into Float32 samples in
 * [-1, 1]. Assumes mono 16-bit PCM (matches the recording config in
 * App.js's Audio.Recording.createAsync call) and reads the actual sample
 * rate from the WAV header rather than assuming 16kHz — if the device
 * recorded at a different rate, resample before calling extractMelSpectrogram
 * or accuracy will be wrong even though nothing throws.
 *
 * @param {string} base64Wav
 * @returns {{ samples: Float32Array, sampleRate: number }}
 */
export function decodeWavBase64(base64Wav) {
  const binary = globalThis.atob
    ? globalThis.atob(base64Wav)
    : Buffer.from(base64Wav, "base64").toString("binary");
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const view = new DataView(bytes.buffer);

  // Standard WAV header: 'RIFF'....'WAVE''fmt '....'data'....<pcm bytes>
  // Locate the 'data' chunk rather than assuming a fixed 44-byte header,
  // since some encoders add extra chunks (e.g. 'LIST') before it.
  const sampleRate = view.getUint32(24, true);
  const bitsPerSample = view.getUint16(34, true);
  let offset = 12;
  let dataOffset = -1, dataSize = 0;
  while (offset < bytes.length - 8) {
    const chunkId = String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
    const chunkSize = view.getUint32(offset + 4, true);
    if (chunkId === "data") {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break;
    }
    offset += 8 + chunkSize + (chunkSize % 2);
  }
  if (dataOffset === -1) throw new Error("decodeWavBase64: no data chunk found");

  const bytesPerSample = bitsPerSample / 8;
  const numSamples = Math.floor(dataSize / bytesPerSample);
  const samples = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    const sampleOffset = dataOffset + i * bytesPerSample;
    if (bitsPerSample === 16) {
      samples[i] = view.getInt16(sampleOffset, true) / 32768;
    } else if (bitsPerSample === 8) {
      samples[i] = (bytes[sampleOffset] - 128) / 128;
    } else {
      throw new Error(`decodeWavBase64: unsupported bit depth ${bitsPerSample}`);
    }
  }
  return { samples, sampleRate };
}

/**
 * @param {Float32Array} pcmSamples - decoded PCM samples, 16kHz mono,
 *   NOT yet normalized or length-fixed (this function does both). Silence
 *   trimming is intentionally NOT applied here — see TODO note above the
 *   melSpectrogramFromSamples doc comment; a fixed 1s window is used as
 *   the v1 shortcut instead.
 * @returns {Float32Array} flattened (N_MELS * nFrames) log-mel, ready to
 *   reshape to the model's [1, N_MELS, nFrames, 1] input.
 */
export function extractMelSpectrogram(pcmSamples) {
  const normalized = normalizeAmplitude(pcmSamples);
  const fixed = fixLength(normalized);
  const logMel = melSpectrogramFromSamples(fixed);
  const nFrames = logMel[0].length;
  const flat = new Float32Array(N_MELS * nFrames);
  let idx = 0;
  for (let m = 0; m < N_MELS; m++) {
    for (let f = 0; f < nFrames; f++) flat[idx++] = logMel[m][f];
  }
  return flat;
}
