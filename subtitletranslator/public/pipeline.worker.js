/**
 * pipeline.worker.js — runs all heavy processing off the main thread.
 * Uses CDN ESM imports (works with credentialless COEP).
 */

import { FFmpeg } from 'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/esm/index.js';
import { toBlobURL } from 'https://cdn.jsdelivr.net/npm/@ffmpeg/util@0.12.1/dist/esm/index.js';
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache  = true;

let ffmpeg = null;

function post(type, payload) { self.postMessage({ type, ...payload }); }
function log(msg) { console.log(`[pipeline] ${msg}`); post('log', { message: msg }); }

function toSRT(segments) {
  return segments.map((s, i) => {
    const fmt = (t) => {
      const ms   = Math.round((t % 1) * 1000);
      const secs = Math.floor(t % 60);
      const mins = Math.floor((t / 60) % 60);
      const hrs  = Math.floor(t / 3600);
      return `${String(hrs).padStart(2,'0')}:${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')},${String(ms).padStart(3,'0')}`;
    };
    return `${i + 1}\n${fmt(s.start)} --> ${fmt(s.end)}\n${s.text.trim()}\n`;
  }).join('\n');
}

function wavToFloat32(wav) {
  let offset = 12;
  while (offset < wav.length - 8) {
    const id = String.fromCharCode(wav[offset], wav[offset+1], wav[offset+2], wav[offset+3]);
    const size = wav[offset+4] | (wav[offset+5] << 8) | (wav[offset+6] << 16) | (wav[offset+7] << 24);
    if (id === 'data') { offset += 8; break; }
    offset += 8 + size;
  }
  const bytesPerSample = 2;
  const numSamples = Math.floor((wav.length - offset) / bytesPerSample);
  const float32 = new Float32Array(numSamples);
  const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  for (let i = 0; i < numSamples; i++) {
    const int16 = view.getInt16(offset + i * 2, true);
    float32[i] = int16 / 32768.0;
  }
  return float32;
}

// ── Map ISO 639-1 codes to Whisper language names ─────────────────────────────
const LANG_CODE_TO_WHISPER = {
  en: 'english',    fr: 'french',     de: 'german',    es: 'spanish',
  it: 'italian',    nl: 'dutch',      ru: 'russian',   zh: 'chinese',
  ar: 'arabic',     hi: 'hindi',      ja: 'japanese',  pt: 'portuguese',
  pl: 'polish',     tr: 'turkish',    ko: 'korean',    sv: 'swedish',
  da: 'danish',     fi: 'finnish',    el: 'greek',     cs: 'czech',
  ro: 'romanian',   hu: 'hungarian',  th: 'thai',      vi: 'vietnamese',
  id: 'indonesian', ms: 'malay',      uk: 'ukrainian', bg: 'bulgarian',
  hr: 'croatian',   sk: 'slovak',     sl: 'slovenian', he: 'hebrew',
  fa: 'persian',    ur: 'urdu',       bn: 'bengali',   ta: 'tamil',
  te: 'telugu',     ml: 'malayalam',  no: 'norwegian', ca: 'catalan',
};

function isCJK(text) {
  return /[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/.test(text);
}

// ── Truncate text at repetition ───────────────────────────────────────────────
function truncateAtRepetition(text) {
  text = text.replace(/([!?.])\1{2,}/g, '$1');
  text = text.replace(/,{2,}/g, ',');
  text = text.replace(/(\s[!?.,:;]){3,}/g, '');

  if (isCJK(text)) {
    const clean = text.replace(/\s/g, '');
    for (let n = 2; n <= 8; n++) {
      for (let i = 0; i <= clean.length - n * 3; i++) {
        const pattern = clean.slice(i, i + n);
        if (/^\s+$/.test(pattern)) continue;
        let reps = 1;
        let pos = i + n;
        while (pos + n <= clean.length && clean.slice(pos, pos + n) === pattern) {
          reps++; pos += n;
        }
        if (reps >= 3) {
          // Find position in original text
          const kept = text.slice(0, text.indexOf(pattern) + pattern.length).trim();
          if (kept.length >= 2) return kept;
        }
      }
    }
    return text.trim();
  }

  const words = text.split(/\s+/);
  if (words.length <= 4) return text.trim();

  for (let phraseLen = 1; phraseLen <= 5; phraseLen++) {
    for (let i = 0; i <= words.length - phraseLen * 2; i++) {
      const phrase = words.slice(i, i + phraseLen).join(' ').toLowerCase().replace(/[^a-z\s]/g, '');
      if (phrase.length < 2) continue;
      let reps = 1;
      let j = i + phraseLen;
      while (j + phraseLen <= words.length) {
        const next = words.slice(j, j + phraseLen).join(' ').toLowerCase().replace(/[^a-z\s]/g, '');
        if (next === phrase) { reps++; j += phraseLen; }
        else break;
      }
      if (reps >= 3) {
        const kept = words.slice(0, i + phraseLen).join(' ').trim();
        if (kept.replace(/[^a-zA-Z]/g, '').length >= 2) return kept;
      }
    }
  }

  return text.trim();
}

// ── Hallucination detection ───────────────────────────────────────────────────
function isHallucination(text) {
  if (!text || text.length === 0) return true;
  if (text.length < 2) return true;

  const meaningful = text.replace(/[\s\p{P}\p{S}\p{N}]/gu, '');
  if (meaningful.length === 0) return true;
  if (meaningful.length < text.length * 0.2) return true;

  if (/^(.)\1+$/.test(text.replace(/\s/g, ''))) return true;

  if (isCJK(text)) {
    const clean = text.replace(/\s/g, '');
    for (let n = 1; n <= 6; n++) {
      if (clean.length < n * 4) continue;
      const counts = {};
      for (let i = 0; i <= clean.length - n; i++) {
        const sub = clean.slice(i, i + n);
        counts[sub] = (counts[sub] || 0) + 1;
      }
      const maxCount = Math.max(...Object.values(counts));
      if (maxCount > (clean.length - n + 1) * 0.4 && maxCount >= 5) return true;
    }
    return false;
  }

  const words = text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/).filter(Boolean);
  if (words.length === 0) return false;
  if (words.length === 1) return false;

  const freq = {};
  for (const w of words) freq[w] = (freq[w] || 0) + 1;
  const maxFreq = Math.max(...Object.values(freq));
  if (words.length > 4 && maxFreq / words.length > 0.5) return true;

  const unique = new Set(words);
  if (unique.size <= 2 && words.length > 8) return true;
  if (unique.size <= 4 && words.length > 25) return true;

  return false;
}

const globalTextCounts = {};
function cleanSegments(segments) {
  return segments
    .map(s => {
      let text = s.text
        .replace(/\[.*?\]/g, '')
        .replace(/\(.*?\)/g, '')
        .replace(/\{.*?\}/g, '')
        .replace(/…+/g, '.')
        .trim();

      text = truncateAtRepetition(text);
      text = text
        .replace(/([!?.])\1{1,}/g, '$1')
        .replace(/,{2,}/g, ',')
        .replace(/\s{2,}/g, ' ')
        .trim();
      text = text.replace(/[\s.,!?;:]+$/, '').trim();

      return { ...s, text };
    })
    .filter(s => s.text.length > 0)
    .filter(s => !isHallucination(s.text))
    .filter((s, i, arr) => i === 0 || s.text.toLowerCase() !== arr[i - 1].text.toLowerCase())
    // Global dedup: limit any identical text to appearing at most 2 times total
    .filter((s) => {
      const key = s.text.toLowerCase().trim();
      if (!globalTextCounts[key]) globalTextCounts[key] = 0;
      globalTextCounts[key]++;
      if (globalTextCounts[key] > 2) {
        log(`  Filtered repeated text (${globalTextCounts[key]}x): "${s.text.slice(0, 60)}"`);
        return false;
      }
      return true;
    })
    .map(s => ({
      ...s,
      text: s.text.length > 250 ? s.text.slice(0, 250).replace(/\s+\S*$/, '').trim() : s.text,
    }));
}

// ── Stage 1: FFmpeg audio extraction ──────────────────────────────────────────
async function extractAudio(fileData) {
  post('progress', { stage: 'ffmpeg', pct: 0, message: 'Loading FFmpeg…' });

  if (!ffmpeg) {
    ffmpeg = new FFmpeg();
    const base = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/esm';
    ffmpeg.on('progress', ({ progress }) =>
      post('progress', { stage: 'ffmpeg', pct: Math.round(progress * 100), message: 'Extracting audio…' })
    );
    await ffmpeg.load({
      coreURL:   await toBlobURL(`${base}/ffmpeg-core.js`,   'text/javascript'),
      wasmURL:   await toBlobURL(`${base}/ffmpeg-core.wasm`, 'application/wasm'),
    });
  }

  await ffmpeg.writeFile('input.video', new Uint8Array(fileData));
  post('progress', { stage: 'ffmpeg', pct: 20, message: 'Extracting audio track…' });

  await ffmpeg.exec(['-i','input.video','-vn','-ar','16000','-ac','1','-f','wav','audio.wav']);

  const wav = await ffmpeg.readFile('audio.wav');
  await ffmpeg.deleteFile('input.video');
  await ffmpeg.deleteFile('audio.wav');

  post('progress', { stage: 'ffmpeg', pct: 100, message: 'Audio extracted.' });
  return wav instanceof Uint8Array ? wav : new Uint8Array(wav);
}

// ── Stage 2: Whisper transcription ────────────────────────────────────────────
const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 30;
const STRIDE_SECONDS = 5;

// Common abbreviations that should NOT trigger a sentence split
const ABBREVIATIONS = /(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|approx|dept|govt|inc|ltd|corp|univ|vol|no|gen|sgt|capt|lt|col|cmdr|adm|pvt|est)$/i;

function splitByPunctuation(segments) {
  const out = [];
  for (const seg of segments) {
    let text = seg.text.trim();
    if (!text) continue;

    const parts = [];
    let remaining = text;
    let lastSplit = 0;

    for (let i = 0; i < remaining.length; i++) {
      const ch = remaining[i];
      if (ch === '\u3002' || ch === '\uff01' || ch === '\uff1f') {
        parts.push(remaining.slice(lastSplit, i + 1).trim());
        lastSplit = i + 1;
        continue;
      }
      if (ch === '.' || ch === '?' || ch === '!') {
        const beforePunc = remaining.slice(lastSplit, i);
        const isAbbreviation = ch === '.' && ABBREVIATIONS.test(beforePunc.split(/\s+/).pop() || '');
        const atEnd = i === remaining.length - 1;
        const followedByNewSentence = i + 1 < remaining.length && remaining[i + 1] === ' ' && i + 2 < remaining.length && /[A-Z\u4e00-\u9fff]/.test(remaining[i + 2]);

        if (!isAbbreviation && (atEnd || followedByNewSentence)) {
          parts.push(remaining.slice(lastSplit, i + 1).trim());
          lastSplit = i + 1;
        }
      }
    }
    const tail = remaining.slice(lastSplit).trim();
    if (tail) parts.push(tail);

    const sentences = parts.filter(p => p.length > 0);

    if (sentences.length <= 1) {
      out.push(seg);
      continue;
    }

    const totalChars = text.length;
    let currentStart = seg.start;
    const duration = seg.end - seg.start;

    for (const sentStr of sentences) {
      const ratio = sentStr.length / totalChars;
      const sentDuration = duration * ratio;
      out.push({
        start: currentStart,
        end: currentStart + sentDuration,
        text: sentStr
      });
      currentStart += sentDuration;
    }
  }
  return out;
}

async function transcribe(wavBytes, model, sourceLang, targetLang, device) {
  post('progress', { stage: 'whisper', pct: 0, message: `Loading ${model}…` });

  // Fix OOM crashes for large models: fp32 requires >3.2GB RAM.
  // Use mixed precision on WebGPU: fp16 encoder (maintains accuracy), q4 decoder (drastically reduces memory).
  const dtype = device === 'webgpu' 
    ? { encoder_model: 'fp16', decoder_model_merged: 'q4' } 
    : 'q8';

  log(`Initializing pipeline with device: "${device}" and dtype: ${JSON.stringify(dtype)}`);

  const asr = await pipeline('automatic-speech-recognition', model, {
    device,
    dtype,
    progress_callback: ({ status, progress }) => {
      if (status === 'downloading')
        post('progress', { stage: 'whisper', pct: Math.round(progress ?? 0), message: 'Downloading Whisper model…' });
    },
  });

  const pcm = wavToFloat32(wavBytes);
  const totalSeconds = pcm.length / SAMPLE_RATE;

  log(`Audio length: ${totalSeconds.toFixed(1)}s (${pcm.length} samples)`);
  post('progress', { stage: 'whisper', pct: 5, message: 'Starting transcription…' });

  // Build Whisper options with native chunking and stride
  const whisperOpts = {
    return_timestamps: true,
    chunk_length_s: CHUNK_SECONDS,
    stride_length_s: STRIDE_SECONDS,
    chunk_callback: (chunk) => {
      // Attempt to provide progress updates if the library version fires this
      const chunkText = (chunk.text ?? '').trim().slice(0, 60);
      const timeStr = chunk.timestamp ? `${chunk.timestamp[0].toFixed(1)}s-${chunk.timestamp[1].toFixed(1)}s` : '...';
      log(`Processing chunk (${timeStr}): "${chunkText}${chunkText.length >= 60 ? '…' : ''}"`);
      post('progress', { stage: 'whisper', pct: 50, message: `Transcribing audio (${timeStr})…` });
    }
  };

  const whisperLang = sourceLang ? LANG_CODE_TO_WHISPER[sourceLang] : null;
  const canWhisperTranslate = model.includes('whisper-small') || model.includes('whisper-large');
  if (whisperLang) {
    whisperOpts.language = whisperLang;
    log(`Whisper language set to: ${whisperLang}`);
    // Only use whisper's built-in translate for capable models (small+).
    // whisper-tiny/base produce garbage translations for non-Latin languages.
    if (targetLang === 'en' && sourceLang !== 'en' && canWhisperTranslate) {
      whisperOpts.task = 'translate';
      log(`Using Whisper built-in translation: ${whisperLang} → english (model is capable)`);
    } else {
      whisperOpts.task = 'transcribe';
      if (targetLang === 'en' && sourceLang !== 'en' && !canWhisperTranslate) {
        log(`Model too small for built-in translate — will transcribe in ${whisperLang} then use OPUS-MT`);
      }
    }
  } else {
    log('No source language specified — Whisper will auto-detect');
  }

  // Pass custom vocabulary hints directly as the prompt if provided
  if (initialPrompt) {
    whisperOpts.prompt = initialPrompt;
    log(`Passing initial prompt: "${initialPrompt}"`);
  }

  log(`Will process ${totalSeconds.toFixed(1)}s audio using native chunking with stride`);
  log(`Whisper options: ${JSON.stringify({ ...whisperOpts, chunk_callback: 'function' })}`);

  post('progress', { stage: 'whisper', pct: 10, message: `Transcribing ${totalSeconds.toFixed(0)}s of audio…` });

  const transcribeStart = performance.now();
  
  // Pass the FULL audio array. The library handles overlapping chunking natively.
  const result = await asr(pcm, whisperOpts);

  const transcribeElapsed = ((performance.now() - transcribeStart) / 1000).toFixed(1);
  
  const rawSegments = (result.chunks ?? [])
    .map(c => ({
      start: c.timestamp?.[0] ?? 0,
      end:   c.timestamp?.[1] ?? 0,
      text:  (c.text ?? '').trim(),
    }))
    .filter(s => s.text.length > 0);

  log(`Whisper finished in ${transcribeElapsed}s — ${rawSegments.length} raw segments extracted`);

  log(`${rawSegments.length} non-empty raw segments`);
  for (let i = 0; i < Math.min(5, rawSegments.length); i++) {
    const s = rawSegments[i];
    log(`  Raw[${i}] ${s.start.toFixed(1)}s–${s.end.toFixed(1)}s: "${s.text.slice(0, 80)}"`);
  }
  if (rawSegments.length > 5) log(`  … and ${rawSegments.length - 5} more raw segments`);

  const cleaned = cleanSegments(rawSegments);

  // Merge adjacent close segments, unless separated by punctuation
  let merged = [];
  for (const seg of cleaned) {
    if (merged.length > 0) {
      const prev = merged[merged.length - 1];
      const gap = seg.start - prev.end;
      const combinedLen = prev.text.length + seg.text.length + 1;
      const prevEndsWithPunc = /[.?!。！？]$/.test(prev.text.trim());
      if (gap < 0.5 && combinedLen < 120 && !prevEndsWithPunc) {
        prev.end = seg.end;
        prev.text = prev.text + ' ' + seg.text;
        continue;
      }
    }
    merged.push({ ...seg });
  }

  // Final pass: ensure any remaining multi-sentence blocks are split
  merged = splitByPunctuation(merged);

  post('progress', { stage: 'whisper', pct: 100, message: `${merged.length} segments transcribed.` });
  return merged;
}

// ── Stage 3: OPUS-MT translation ──────────────────────────────────────────────
async function translate(segments, model, device) {
  if (segments.length === 0) return segments;

  post('progress', { stage: 'translate', pct: 0, message: `Loading ${model}…` });

  const dtype = device === 'webgpu' ? 'fp32' : 'q8';
  const translator = await pipeline('translation', model, {
    device,
    dtype,
    progress_callback: ({ status, progress }) => {
      if (status === 'downloading')
        post('progress', { stage: 'translate', pct: Math.round(progress ?? 0), message: 'Downloading translation model…' });
    },
  });

  post('progress', { stage: 'translate', pct: 10, message: `Translating ${segments.length} segments…` });

  const BATCH_SIZE = 10;
  const out = [];
  const totalBatches = Math.ceil(segments.length / BATCH_SIZE);

  for (let batch = 0; batch < totalBatches; batch++) {
    const start = batch * BATCH_SIZE;
    const end = Math.min(start + BATCH_SIZE, segments.length);
    const batchSegments = segments.slice(start, end);
    const texts = batchSegments.map(s => s.text.trim());

    const pct = Math.round(10 + ((batch / totalBatches) * 85));
    post('progress', { stage: 'translate', pct, message: `Translating batch ${batch + 1}/${totalBatches}…` });

    const results = await translator(texts);
    for (let i = 0; i < batchSegments.length; i++) {
      out.push({ ...batchSegments[i], text: results[i]?.translation_text ?? batchSegments[i].text });
    }
  }

  post('progress', { stage: 'translate', pct: 100, message: 'Translation complete.' });
  return out;
}

// ── Entry point ────────────────────────────────────────────────────────────────
self.onmessage = async ({ data }) => {
  const { fileData, whisperModel, replacements, opusModel, skipTranslation, sourceLang, targetLang, device } = data;
  try {
    const audio = await extractAudio(fileData);

    const canWhisperTranslate = whisperModel.includes('whisper-small') || whisperModel.includes('whisper-large');
    const whisperWillTranslate = canWhisperTranslate && sourceLang && sourceLang !== 'en' && targetLang === 'en';

    // Reset global dedup counts for each run
    for (const key in globalTextCounts) delete globalTextCounts[key];

    const segments = await transcribe(audio, whisperModel, sourceLang, targetLang, device);

    if (segments.length === 0) {
      post('error', { message: 'No speech detected in the audio. The file may contain only non-speech sounds (music, noise, etc.), or the audio quality is too low for Whisper to detect speech. Try a different Whisper model or check that the file has a clear speech track.' });
      return;
    }

    let final;
    if (whisperWillTranslate) {
      log('Whisper already translated to English — skipping OPUS-MT');
      final = segments;
    } else if (!skipTranslation && opusModel) {
      final = await translate(segments, opusModel, device);
    } else {
      final = segments;
    }

    // Apply word replacements (post-processing)
    if (replacements && replacements.length > 0) {
      log(`Applying ${replacements.length} word replacement(s)…`);
      for (const seg of final) {
        for (const { from, to } of replacements) {
          const regex = new RegExp(from.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
          const before = seg.text;
          seg.text = seg.text.replace(regex, to);
          if (before !== seg.text) {
            log(`  Replaced "${from}" → "${to}" in: "${seg.text.slice(0, 60)}"`);
          }
        }
      }
    }

    post('done', { srt: toSRT(final), segments: final });
  } catch (err) {
    post('error', { message: err?.message ?? String(err) });
  }
};
