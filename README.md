# SubtitleAI — Local Video Subtitle Maker & Translator

100% client-side AI tool that generates and translates subtitles for any video or audio file. No server, no API key, no data leaving your machine.

## What It Does

Drop a video or audio file and get back a clean `.srt` subtitle file. The entire pipeline runs in your browser:

1. **FFmpeg (WASM)** extracts audio from video files
2. **OpenAI Whisper** transcribes speech to text (4 model sizes: 40MB → 800MB)
3. **Translation** across 35+ languages using M2M100, NLLB-200, or OPUS-MT
4. **Hallucination cleanup** automatically filters repeated/garbage segments
5. **SRT generation** with proper timestamps and formatting

## Three Compute Modes

### CPU (WASM)
- Works on any device with a modern browser
- Uses ONNX Runtime WebAssembly backend
- Slower but universal — no GPU required
- Runs Whisper with `q8` quantization

### GPU (WebGPU)
- Significant speedup for large models
- Mixed precision: `fp16` encoder + `q4` decoder (Whisper)
- Requires browser WebGPU support and ~1.5GB+ VRAM
- Translation models run at `q4` (M2M100) or `q4f16` (NLLB-200)

### Cloud API
- Plug in any OpenAI-compatible endpoint (Groq, Colab, etc.)
- Audio leaves your machine, processed server-side
- Network retry logic with offline recovery
- Saves API settings to localStorage

## Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `whisper-tiny` | ~40 MB | Fastest | Basic |
| `whisper-base` | ~150 MB | Balanced | Good |
| `whisper-small` | ~250 MB | Moderate | Accurate |
| `whisper-large-v3-turbo` | ~800 MB | Slow | Highly accurate |

The `small` and `large` models support Whisper's built-in translation task (to English). Smaller models transcribe in the source language, then use an external translation model.

## Translation Models

| Model | Languages | Size | Device |
|-------|-----------|------|--------|
| **M2M100 418M** (Recommended) | 100 | ~240 MB | WebGPU / CPU |
| **NLLB-200 Distilled 600M** | 200+ | ~889 MB | WebGPU (needs 1.5GB+ VRAM) |
| **OPUS-MT** | Pair-specific | Lightweight | CPU only |

### Supported Languages (35+)

English, French, German, Spanish, Italian, Dutch, Russian, Chinese, Arabic, Hindi, Japanese, Portuguese, Polish, Turkish, Korean, Thai, Vietnamese, Indonesian, Ukrainian, Czech, Swedish, Danish, Finnish, Greek, Romanian, Hungarian, Norwegian, Bengali, Tamil, Telugu, Malayalam, Swahili, Hebrew, Persian, Urdu.

## Hallucination Cleanup

Whisper is known to hallucinate on silent segments (repeating "Thank you for watching", etc.). The cleanup pipeline handles this:

- **Repetition detection**: N-gram analysis at word and character level, CJK-aware
- **Hallucination filtering**: Detects single-character repeats, punctuation garbage, low-meaningful-content segments
- **Global deduplication**: Limits any identical text to appearing at most 2 times
- **Sentence deduplication**: Catches translation model sentence loops (common with OPUS-MT)
- **Punctuation cleanup**: Collapses runs of repeated punctuation, strips artifacts

## Word Corrections

Fix consistently mistranscribed words post-generation. Enter corrections in the textarea using `→` or `->` as separator:

```
probly → probably, gonna → going to
```

Corrections are applied via regex (case-insensitive) across all subtitle segments.

## Architecture

```
subtitlemakertranslator/
├── src/
│   ├── pages/index.astro        # Main UI + client-side orchestration
│   ├── layouts/Layout.astro     # HTML shell
│   ├── styles/global.css        # Tailwind theme (green accent)
│   └── components/Welcome.astro # Default Astro component
├── public/
│   ├── inference.worker.js      # Web Worker: Whisper + Translation
│   └── pipeline.worker.js       # Web Worker: FFmpeg + Whisper + Translation
└── dist/                        # Built output (gitignored)
```

### Worker Architecture

- **Main thread** (`index.astro`): UI, FFmpeg audio extraction, file I/O
- **Inference worker** (`inference.worker.js`): Runs Whisper transcription and translation models off the main thread to keep UI responsive
- FFmpeg stays on the main thread to avoid COEP/Worker compatibility issues

### Key Technical Details

- Audio is extracted at 16kHz mono WAV
- Whisper processes audio in 30-second chunks with 5-second stride overlap
- Sentence splitting respects abbreviations (Mr., Dr., etc.) and CJK punctuation
- M2M100 uses `src_lang`/`tgt_lang` options with `max_new_tokens: 64`
- NLLB-200 requires `>>tgt_lang<<` prefix on each input text
- Models are cached in browser via HuggingFace Transformers cache

## Performance

- A 3-minute video on local GPU takes ~1.5 minutes
- Audio input is faster than video (skips FFmpeg extraction)
- Larger models are more accurate but slower
- WebGPU mixed precision (fp16 encoder + q4 decoder) reduces memory while maintaining accuracy

## Getting Started

```sh
npm install
npm run dev
```

Open `localhost:4321` in your browser.

### Build for Production

```sh
npm run build
npm run preview
```

## Tech Stack

- **Astro** — Framework
- **Tailwind CSS** — Styling
- **HuggingFace Transformers.js** — ML model inference
- **ONNX Runtime** — WebAssembly / WebGPU backend
- **FFmpeg WASM** — Browser-side video/audio processing

## Limitations

- WebGPU not supported in all browsers (Chrome/Edge recommended)
- Large models (NLLB-200, whisper-large) need significant VRAM
- Whisper can still hallucinate on music-only or noisy segments
- No subtitle editing UI — outputs raw SRT
- No video preview or timeline scrubbing
- Browser memory caps limit max model size
