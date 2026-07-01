# Sinestesia Musical v2 — Research Report: Multi-Engine Separation + Multi-Spiral

Consolidated from dedicated deep-research agents (2026-06-30). Every figure is cited.
"SDR" = signal-to-distortion ratio on MUSDB18(-HQ) unless noted; higher = better.
Companion to the execution plan (multi-engine separation + one spiral per component,
colored by log-frequency→hue, with a toggle expanding 1→N spirals).

## Goal & constraints
Split a song into components (vocals / drums / bass / keys / melody …), track each
component's pitch, and draw one spiral per component. Multiple **user-selectable
separation engines**, from free/instant classical DSP to self-hosted ONNX ML.
Constraints: current app is Flask on Render **free tier (512 MB, ~0.1 vCPU, no GPU)**;
analysis runs in an isolated subprocess using **numpy + scipy + ffmpeg** (no librosa);
output is precomputed per-frame time-series consumed by a `<canvas>` renderer.

**Decisions locked:** ML backend = **self-hosted ONNX** (MDX-Net / SCNet, torch-free);
build a **framework + multiple switchable engines**; color = **log-frequency→hue**;
visual = **toggle 1→N spirals**.

## Engine spectrum (what to build)
| Tier | Engine | Separates | Runs where | Quality | Cost |
|---|---|---|---|---|---|
| Free/instant | **bands** | energy proxies (kick/vocal-body/hats) | inline numpy | proxy only | $0 |
| Free/instant | **HPSS** | harmonic (melody/voice/keys) vs percussive (drums) | inline scipy | good pitched↔drums; poor pure-vocal | $0 |
| Free/cheap | **REPET-SIM** | repeating background vs **vocal** | inline numpy (block-wise) | low-single-digit SDR; ≪ ML | $0 |
| Free (stereo) | **ADRess / mid-side** | sources by stereo pan (center=vocal) | inline numpy | good on clean pan-pot mixes | $0 |
| Free (abstract) | **NMF** | K unlabeled components | inline numpy | poor named stems; nice visual curves | $0 |
| **HD (chosen)** | **ONNX MDX/SCNet** | vocals / 4-stem, torch-free | self-hosted onnxruntime | vocals ~9–9.9 dB | $0 compute, RAM/time TBD |
| Fallback (not chosen) | Demucs serverless GPU / turnkey API | 4–6 stems ~9.2–13.5 dB | Modal/Replicate / Music.ai | best | ~$0.01–0.28/song |

---

## 1. HPSS — Harmonic/Percussive via median filtering
Fitzgerald 2010 + Driedger 2014. Median-filter magnitude spectrogram **along time**
→ harmonic-enhanced `Henh`; **along frequency** → percussive-enhanced `Penh`.
- Fitzgerald soft mask: `MH = Henh^p/(Henh^p+Penh^p)` (p=2), kernel 17, FFT 4096 / hop 1024.
- Driedger β-masks + residual: `Mh = Henh/Penh > β`, `Mp = Penh/Henh ≥ β`, `Mr = 1−Mh−Mp`;
  β=2; kernels ≈ 200 ms (time) & 500 Hz (freq).
- Memory: only the time-median needs history → **overlapping time blocks** (overlap =
  half the time-kernel), `scipy.ndimage.median_filter(size=(1,Lt))` and `(Lf,1)`.
- Quality: strong pitched↔drums; **vocals ambiguous** (sustained→harmonic, consonants→
  percussive) → poor pure-vocal isolator.
- Sources: https://dafx10.iem.at/papers/DerryFitzGerald_DAFx10_P15.pdf ·
  https://www.audiolabs-erlangen.de/resources/2014-ISMIR-ExtHPSep/2014_DriedgerMuellerDisch_ExtensionsHPSeparation_ISMIR.pdf ·
  https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S1_HRPS.html

## 2. REPET-SIM — repeating background vs vocal foreground
Rafii & Pardo. Self-similarity matrix `S(a,b)=cosine(V[:,a],V[:,b])`; per frame take median
of its top-k most-similar frames → repeating model `W`; mask `M=min(W,V)/V` → background;
**voice = mixture − background**. Defaults: N=2048 Hamming half-overlap, k=100, t=0, d=1 s,
voice high-pass 100 Hz.
- Memory: T×T matrix is the risk (~328 MB @hop1024/44k). Fix: **row-blocks**
  (`Vn[:,block].T @ Vn`, keep top-k, discard) + `hop≥1024`/`sr=22050` → ~80 MB.
- Pure numpy/scipy reference: `zafarrafii/REPET-Python` (no librosa). Quality: low-single-
  digit SDR; good for pop/rock, bad for jazz/dense/reverb.
- Sources: https://users.cs.northwestern.edu/~zra446/doc/Rafii-Pardo%20-%20Music-Voice%20Separation%20using%20the%20Similarity%20Matrix%20-%20ISMIR%202012.pdf ·
  https://github.com/zafarrafii/REPET-Python

## 3. Stereo — ADRess & mid-side (needs stereo decode)
- **ADRess** (Barry 2004): per frame `AzR(k,i)=|Lf−g(i)·Rf|`, g(i)=i/β, β=100; nulls→peaks
  give a **per-frame azimuth-energy distribution** (energy vs pan) — itself spiral-ready
  (center ≈ vocals/bass/kick). FFT 4096 / hop 1024. Fails on reverb / M-S / off-center.
- **Mid-side / OOPS:** `Mid=(L+R)/2`, `Side=(L−R)/2`; `L−R` cancels center (karaoke). Trivial.
  Freq-domain center mask `|Lf·conj(Rf)|/((|Lf|²+|Rf|²)/2)` is cleaner.
- Source: https://arrow.tudublin.ie/argcon/21/

## 4. NMF (optional abstract mode)
`V≈WH`; Lee-Seung KL multiplicative updates, K=2–8, ~150 iters, hand-rolled numpy (avoid
sklearn dep). Components **unlabeled** (poor named stems) but each activation row `H[k,:]`
is a ready-made per-frame curve → nice "K abstract spirals". ~110 MB / ~1–3 min @0.1 vCPU.
Source: https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization

## 5. Bands (default, can't fail)
STFT bin-sum energy per band + per-band centroid. Zones: sub 20–120, bass 120–300, low-mid
300–800, mid 800–3k, presence 3–6k, air 6–16k Hz. Not separation, but robust always-on
proxies. Source: https://www.izotope.com/en/learn/eq-cheat-sheet

## 6. ML separation — chosen self-hosted ONNX path (+ fallbacks)
- **Render has NO GPU; free/Starter (512 MB) OOMs PyTorch Demucs** (~1.5–4 GB/song).
  https://render.com/pricing · https://github.com/facebookresearch/demucs
- **Chosen: torch-free ONNX.** onnxruntime CPU wheel ~19 MB vs torch ~532 MB.
  - **MDX-Net** (KUIELab): vocals SDR ~9.0; UVR ONNX models 30–67 MB; torch-free runner
    `seanghay/uvr-mdx-infer`; weights https://huggingface.co/seanghay/uvr_models
  - **SCNet ONNX**: 4-stem, vocals ~9.9 dB, FP16 22.6 MB; torch-free CPU proof (WASM 2.83×
    real-time) `elicwhite/scnet-web-wasm`
  - **Open gap (measure first):** peak RSS + wall-time on a weak 1–2 vCPU box are unpublished.
- **Fallbacks (documented, not chosen):**
  - Serverless GPU Demucs + cache: Replicate `cjwbw/demucs` $0.02/song; Modal T4 ~$0.005–
    0.015/song ($30/mo free credit ≈ 2–3k songs); RunPod/Beam ~$0.01–0.03. 4 stems ~9.2 dB.
  - Turnkey APIs: Music.ai $0.05–0.15/min; LALAL.ai ~$0.06/min; Fadr $10/mo+$0.05/min;
    AudioShake best (13.5 dB, enterprise); Gaudio $0.065/min.
- Demucs quality ref: htdemucs 8.8 / htdemucs_ft 9.0–9.2; RoFormers 11–12.7 dB (GPU-only).
  https://arxiv.org/abs/2211.08553

## 7. Frontend renderer map (`templates/index.html`)
- State `freqTimes/freqValues` ~L601-602; `applyFrequencyData` ~L649-664; `pollJob` ~L837-859; `/upload` ~L740.
- `drawSpiral(freq)` ~L474-562 (log spiral `r=16.5·e^0.327φ`, canvas `#spiral-canvas` 480×480,
  clear ~L484, hue=(φ mod 2π)/2π ~L521, `toXY`, `hsvToRgb` ~L440); `freqAtTime` ~L612-627;
  loop `frame()` ~L675-694.
- Multi-spiral: globals→array; split `drawSpiral` into once/frame background + per-series layer;
  alpha grading + slight per-layer rotation.

## 8. Color + layout
- **Log-frequency→hue:** `hue = (12*log2(f/55)*30) % 360`, HSL, S 70–100%, L 40–60% (octaves
  wrap). https://delu.medium.com/a-perceptually-meaningful-audio-visualizer-ee72051781bc
- Legibility for N layers: ≤4 hue zones, opacity grading, soft trail (`fillRect rgba(bg,0.05)`),
  per-layer rotation. Refs: MilkDrop/projectM, Sonic Visualiser, https://photosounder.com/spiral/

## 9. Caching / deploy (Phase 5)
- Cache **derived analysis only** (~1 MB `.npz`) not audio (~170 MB WAV) → 170× smaller.
- **Cloudflare R2**: 10 GB free, **zero egress**. Key = `{video_id}/{engine}/{model_version}/{params_hash}`.
- Render **one-off Jobs** bill per-second (~$0.01–0.02/song CPU) if separation is offloaded.
- https://developers.cloudflare.com/r2/pricing/ · https://render.com/docs/jobs
