/**
 * inference.worker.js — handles only Whisper + Translation (the slow parts).
 * FFmpeg stays on the main thread to avoid COEP/Worker issues.
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache  = true;
env.backends.onnx.wasm.numThreads = 2; // maintain low memory

function post(type, payload) { self.postMessage({ type, ...payload }); }
function log(msg) { console.log(`[worker] ${msg}`); post('log', { message: msg }); }

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

const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 30;
const STRIDE_SECONDS = 5;

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

// ── Detect if text contains CJK / non-space-delimited characters ──────────────
function isCJK(text) {
  return /[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/.test(text);
}

// ── Truncate text at the point where repetition starts ────────────────────────
function truncateAtRepetition(text) {
  // 1. Collapse runs of the same punctuation character: !!!!! → !, ..... → .
  text = text.replace(/([!?.])\1{2,}/g, '$1');
  text = text.replace(/,{2,}/g, ',');
  // Remove isolated single-char punctuation noise: . . . . or ! ! ! !
  text = text.replace(/(\s[!?.,:;]){3,}/g, '');

  // Collapse alternating punctuation hallucinations: ?.?.?.? → ?, !.!.! → !
  text = text.replace(/([!?.,;])\1?(?:\s*[!?.,;]\1?){2,}/g, '$1');

  // Collapse character-level stutters/hallucinations (e.g. ssssssss -> ss, essessess -> ess)
  text = text.replace(/([a-zA-Z])\1{4,}/gi, '$1$1');
  text = text.replace(/([a-zA-Z]{2,5})\1{3,}/gi, '$1');

  // Strip trailing/leading punctuation noise (OPUS-MT artifacts like "?.?.?.?.?")
  text = text.replace(/[\s!?.,;:]{3,}$/g, '').trim();
  text = text.replace(/^[\s!?.,;:]{3,}/g, '').trim();

  // If nothing meaningful left, return empty
  if (text.replace(/[^a-zA-Z0-9\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]/g, '').length < 2) return '';

  // 2. For CJK text: detect character-level n-gram repetition
  if (isCJK(text)) {
    // Check for repeated character sequences of length 2-8
    for (let n = 2; n <= 8; n++) {
      for (let i = 0; i <= text.length - n * 3; i++) {
        const pattern = text.slice(i, i + n);
        if (/^\s+$/.test(pattern)) continue; // skip whitespace
        let reps = 1;
        let pos = i + n;
        while (pos + n <= text.length && text.slice(pos, pos + n) === pattern) {
          reps++;
          pos += n;
        }
        if (reps >= 3) {
          const kept = text.slice(0, i + n).trim();
          if (kept.length >= 2) return kept;
        }
      }
    }
    return text.trim();
  }

  // 3. For space-delimited text: detect word-level repetition
  const words = text.split(/\s+/);
  if (words.length <= 4) return text.trim();

  for (let phraseLen = 1; phraseLen <= 12; phraseLen++) {
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

  // Detect period/comma separated repetition: "X. X. X." or "X, X, X"
  // Strip punctuation and split into phrases
  const phrases = text.split(/[.,;]\s*/).map(s => s.replace(/[^a-zA-Z\s]/g, '').trim().toLowerCase()).filter(Boolean);
  if (phrases.length >= 3) {
    const freq = {};
    for (const p of phrases) freq[p] = (freq[p] || 0) + 1;
    const maxFreq = Math.max(...Object.values(freq));
    if (maxFreq >= 3 && maxFreq / phrases.length > 0.3) {
      const seen = new Set();
      const kept = [];
      for (const p of phrases) {
        if (seen.has(p)) break;
        seen.add(p);
        kept.push(p);
      }
      const result = kept.join(', ').trim();
      if (result.length >= 3) return result;
    }
  }

  return text.trim();
}

// ── Post-processing: detect hallucinated segments ─────────────────────────────
function isHallucination(text) {
  if (!text || text.length === 0) return true;

  const stripped = text.replace(/\s/g, '');

  // Single repeated character (e.g., "ssssssssss")
  if (/^(.)\1+$/.test(stripped)) return true;

  // Alternating punctuation garbage (e.g., "?.?.?.?", "!.!.!")
  if (/^([!?.,;])\1?(?:\s*\1?\s*){3,}$/.test(stripped)) return true;
  if (/^[\s!?.,;:]+$/.test(text)) return true;

  // Only punctuation/special chars, no real words
  if (text.replace(/[^a-zA-Z\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\s]/g, '').trim().length === 0) return true;

  // Very short text that's mostly punctuation (e.g., "? ? ?")
  const alphaRatio = text.replace(/[^a-zA-Z\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]/g, '').length / text.length;
  if (text.length > 2 && alphaRatio < 0.2) return true;

  return false;
}

function deduplicateSentences(text) {
  // Split into sentences on common delimiters
  const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 0);
  if (sentences.length <= 2) return text;

  const seen = new Set();
  const kept = [];
  for (const sent of sentences) {
    const key = sent.toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, ' ').trim();
    if (key.length < 3) { kept.push(sent); continue; }
    if (seen.has(key)) continue;
    seen.add(key);
    kept.push(sent);
  }
  return kept.join(' ').trim();
}

function cleanSegments(segments) {
  const before = segments.length;
  const result = segments
    .map(s => {
      // Step 1: Strip annotations
      let text = s.text
        .replace(/\[.*?\]/g, '')
        .replace(/\(.*?\)/g, '')
        .replace(/\{.*?\}/g, '')
        .replace(/…+/g, '.')
        .trim();

      // Step 2: Truncate at repetition
      text = truncateAtRepetition(text);

      // Step 3: Deduplicate sentences within segment (catches OPUS-MT sentence loops)
      text = deduplicateSentences(text);

      // Step 4: Final punctuation cleanup
      text = text
        .replace(/([!?.])\1{1,}/g, '$1')
        .replace(/,{2,}/g, ',')
        .replace(/\s{2,}/g, ' ')
        .replace(/^[\s.!?,:;]+/, '')
        .replace(/[\s.!?,:;]+$/, '')
        .trim();

      return { ...s, text };
    })
    // Remove empty
    .filter(s => s.text.length > 0)
    // Remove hallucinations
    .filter(s => {
      const hall = isHallucination(s.text);
      if (hall) log(`  Filtered hallucination: "${s.text.slice(0, 80)}…"`);
      return !hall;
    })
    // Remove consecutive duplicates (case-insensitive)
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
    // Cap length (safety net — break at word boundary)
    .map(s => ({
      ...s,
      text: s.text.length > 250 ? s.text.slice(0, 250).replace(/\s+\S*$/, '').trim() : s.text,
    }));

  log(`Cleanup: ${before} → ${result.length} segments (removed ${before - result.length})`);
  return result;
}
const globalTextCounts = {};

// Common abbreviations that should NOT trigger a sentence split
const ABBREVIATIONS = /(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|approx|dept|govt|inc|ltd|corp|univ|vol|no|gen|sgt|capt|lt|col|cmdr|adm|pvt|est)$/i;

function splitByPunctuation(segments) {
  const out = [];
  for (const seg of segments) {
    let text = seg.text.trim();
    if (!text) continue;

    // Split on sentence-ending punctuation followed by a space and uppercase letter,
    // or CJK sentence terminators, or punctuation at end of string.
    // This avoids splitting on abbreviations like "Mr." or "Dr."
    const parts = [];
    let remaining = text;
    let lastSplit = 0;

    for (let i = 0; i < remaining.length; i++) {
      const ch = remaining[i];
      // CJK sentence terminators always split
      if (ch === '\u3002' || ch === '\uff01' || ch === '\uff1f') {
        parts.push(remaining.slice(lastSplit, i + 1).trim());
        lastSplit = i + 1;
        continue;
      }
      // Latin sentence terminators: split only if followed by space+uppercase or end of string
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
    // Push any remaining text
    const tail = remaining.slice(lastSplit).trim();
    if (tail) parts.push(tail);

    // Filter empty
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

async function transcribe(pcmFloat32, model, sourceLang, targetLang, device) {
  log(`Loading Whisper model: ${model}`);
  post('progress', { stage: 'whisper', pct: 0, message: `Loading ${model}…` });

  // Fix OOM crashes for large models: fp32 requires >3.2GB RAM.
  // Use mixed precision on WebGPU: fp16 encoder (maintains accuracy), q4 decoder (drastically reduces memory).
  const dtype = device === 'webgpu' 
    ? { encoder_model: 'fp16', decoder_model_merged: 'q4' } 
    : 'q8';

  log(`Initializing pipeline with device: "${device}" and dtype: ${JSON.stringify(dtype)}`);

  let asr;
  try {
    asr = await pipeline('automatic-speech-recognition', model, {
      device,
      dtype,
      progress_callback: ({ status, progress }) => {
        if (status === 'downloading') {
          const p = Math.round(progress ?? 0);
          log(`Downloading Whisper model: ${p}%`);
          post('progress', { stage: 'whisper', pct: p, message: `Downloading Whisper model… ${p}%` });
        }
      },
    });
  } catch (modelErr) {
    log(`Failed to load model on ${device}: ${modelErr.message}`);
    // If WebGPU fails, try falling back to WASM
    if (device === 'webgpu') {
      log('WebGPU model load failed — retrying with WASM/CPU…');
      post('progress', { stage: 'whisper', pct: 0, message: 'GPU failed, falling back to CPU…' });
      asr = await pipeline('automatic-speech-recognition', model, {
        device: 'wasm',
        dtype: 'q8',
        progress_callback: ({ status, progress }) => {
          if (status === 'downloading') {
            const p = Math.round(progress ?? 0);
            post('progress', { stage: 'whisper', pct: p, message: `Downloading Whisper model… ${p}%` });
          }
        },
      });
    } else {
      throw modelErr;
    }
  }

  log('Whisper model loaded. Starting transcription…');
  post('progress', { stage: 'whisper', pct: 5, message: 'Starting transcription…' });

  const totalSeconds = pcmFloat32.length / SAMPLE_RATE;
  log(`Audio length: ${totalSeconds.toFixed(1)}s (${pcmFloat32.length} samples)`);

  if (pcmFloat32.length < SAMPLE_RATE) {
    throw new Error('Audio is too short (less than 1 second). Please upload a longer file.');
  }

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

  // Set language if specified
  const whisperLang = sourceLang ? LANG_CODE_TO_WHISPER[sourceLang] : null;
  const canWhisperTranslate = model.includes('whisper-small') || model.includes('whisper-large');
  if (whisperLang) {
    whisperOpts.language = whisperLang;
    log(`Whisper language set to: ${whisperLang}`);

    // Only use whisper's built-in translate for capable models (small+).
    // whisper-tiny/base produce garbage translations (hallucinated English).
    // For smaller models, transcribe in source language → OPUS-MT translates.
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

  log(`Will process ${totalSeconds.toFixed(1)}s audio using native chunking with stride`);
  log(`Whisper options: ${JSON.stringify({ ...whisperOpts, chunk_callback: 'function' })}`);

  post('progress', { stage: 'whisper', pct: 10, message: `Transcribing ${totalSeconds.toFixed(0)}s of audio…` });

  const transcribeStart = performance.now();
  
  // Pass the FULL audio array. The library handles overlapping chunking natively.
  const result = await asr(pcmFloat32, whisperOpts);

  const transcribeElapsed = ((performance.now() - transcribeStart) / 1000).toFixed(1);
  
  const rawSegments = (result.chunks ?? [])
    .map(c => ({
      start: c.timestamp?.[0] ?? 0,
      end:   c.timestamp?.[1] ?? 0,
      text:  (c.text ?? '').trim(),
    }))
    .filter(s => s.text.length > 0);

  log(`Whisper finished in ${transcribeElapsed}s — ${rawSegments.length} raw segments extracted`);
  
  if (rawSegments.length === 0) {
    log(`Result object keys: ${Object.keys(result)}`);
    log(`Result text: "${(result.text ?? '').slice(0, 200)}"`);
    log(`Result chunks count: ${(result.chunks ?? []).length}`);
    // If result has text but no chunks, it might be a model output format issue
    if (result.text && result.text.trim().length > 0) {
      log('Whisper returned text but no timestamped chunks — using full text as single segment');
      rawSegments.push({ start: 0, end: totalSeconds, text: result.text.trim() });
    }
  }

  // Log first 10 raw segments for debugging
  for (let i = 0; i < Math.min(10, rawSegments.length); i++) {
    const s = rawSegments[i];
    log(`  Raw[${i}] ${s.start.toFixed(1)}s–${s.end.toFixed(1)}s: "${s.text.slice(0, 100)}${s.text.length > 100 ? '…' : ''}"`);
  }
  if (rawSegments.length > 10) log(`  … and ${rawSegments.length - 10} more raw segments`);

  log('Running cleanup pipeline…');
  const cleaned = cleanSegments(rawSegments);

  // Merge adjacent segments that are very close in time, unless separated by punctuation
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

  log(`After merging: ${merged.length} final subtitle segments`);
  // Log first 10 final segments
  for (let i = 0; i < Math.min(10, merged.length); i++) {
    const s = merged[i];
    log(`  Final[${i}] ${s.start.toFixed(1)}s–${s.end.toFixed(1)}s: "${s.text.slice(0, 100)}${s.text.length > 100 ? '…' : ''}"`);
  }
  if (merged.length > 10) log(`  … and ${merged.length - 10} more segments`);

  post('progress', { stage: 'whisper', pct: 100, message: `${merged.length} segments transcribed.` });
  return merged;
}

async function translate(segments, translatorConfig, device) {
  if (segments.length === 0 || !translatorConfig) return segments;

  const { type, model, srcLang, tgtLang } = translatorConfig;
  const isNLLB = type === 'nllb';
  const isM2M100 = type === 'm2m100';

  const modelNameDesc = isM2M100 ? 'M2M100' : (isNLLB ? 'NLLB-200' : 'OPUS-MT');
  log(`Loading translation model: ${model} (${modelNameDesc})`);
  post('progress', { stage: 'translate', pct: 0, message: `Loading ${model}…` });

  let dtype;
  if (isM2M100) dtype = 'q4';
  else if (isNLLB) dtype = (device === 'webgpu' ? 'q4f16' : 'q8');
  else dtype = (device === 'webgpu' ? 'fp32' : 'q8');
  const translator = await pipeline('translation', model, {
    device,
    dtype,
    progress_callback: ({ status, progress }) => {
      if (status === 'downloading') {
        const p = Math.round(progress ?? 0);
        log(`Downloading translation model: ${p}%`);
        post('progress', { stage: 'translate', pct: p, message: `Downloading translation model… ${p}%` });
      }
    },
  });

  log(`Translation model loaded. Translating ${segments.length} segments…`);
  post('progress', { stage: 'translate', pct: 10, message: `Translating ${segments.length} segments…` });

  const BATCH_SIZE = (isNLLB || isM2M100) ? 5 : 10;
  const out = [];
  const totalBatches = Math.ceil(segments.length / BATCH_SIZE);

  for (let batch = 0; batch < totalBatches; batch++) {
    const start = batch * BATCH_SIZE;
    const end = Math.min(start + BATCH_SIZE, segments.length);
    const batchSegments = segments.slice(start, end);

    // NLLB-200 requires >>tgt_lang<< prefix on each input text
    const texts = batchSegments.map(s => {
      const text = s.text.trim();
      return isNLLB ? `>>${tgtLang}<< ${text}` : text;
    });

    log(`Translating batch ${batch + 1}/${totalBatches} (${start + 1}–${end} of ${segments.length})…`);
    post('progress', {
      stage: 'translate',
      pct: Math.round(10 + ((batch / totalBatches) * 85)),
      message: `Translating batch ${batch + 1}/${totalBatches}…`,
    });

    const options = isM2M100 ? { src_lang: srcLang, tgt_lang: tgtLang, max_new_tokens: 64 } : {};
    const results = await translator(texts, options);
    for (let i = 0; i < batchSegments.length; i++) {
      out.push({ ...batchSegments[i], text: results[i]?.translation_text ?? batchSegments[i].text });
    }

    log(`Batch ${batch + 1}/${totalBatches} done — ${out.length}/${segments.length} translated`);
  }

  log(`Translation complete: ${out.length} segments`);
  post('progress', { stage: 'translate', pct: 100, message: 'Translation complete.' });
  return out;
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'post_process') {
      const { segments, translatorConfig, replacements, device } = data;
      log(`Received ${segments.length} segments for post-processing.`);
      for (const key in globalTextCounts) delete globalTextCounts[key];

      // Clean raw whisper hallucinations from Cloud API
      let cleaned = cleanSegments(segments);

      let final = cleaned;
      if (translatorConfig) {
        final = await translate(cleaned, translatorConfig, device);
        // Clean again in case translation model hallucinates
        final = cleanSegments(final);
      }

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
      return;
    }

    const { pcm, whisperModel, replacements, translatorConfig, skipTranslation, sourceLang, targetLang, device } = data;
    const float32 = new Float32Array(pcm);
    log(`Received PCM: ${float32.length} samples (${(float32.length / SAMPLE_RATE).toFixed(1)}s)`);
    log(`Source language: ${sourceLang || 'auto-detect'}, Target language: ${targetLang || 'none'}`);

    // whisper-small and whisper-large are capable of built-in translate;
    // tiny/base should transcribe in source language and use translation model.
    const canWhisperTranslate = whisperModel.includes('whisper-small') || whisperModel.includes('whisper-large');
    const whisperWillTranslate = canWhisperTranslate && sourceLang && sourceLang !== 'en' && targetLang === 'en';

    // Reset global dedup counts for each run
    for (const key in globalTextCounts) delete globalTextCounts[key];

    const segments = await transcribe(float32, whisperModel, sourceLang, targetLang, device);
    if (segments.length === 0) {
      log('Transcription returned 0 segments after cleanup');
      log(`Original raw segments were filtered out by hallucination detection or cleanup`);
      post('error', { message: 'No speech detected in the audio. The file may contain only non-speech sounds (music, noise, etc.), or the audio quality is too low for Whisper to detect speech. Try a different Whisper model or check that the file has a clear speech track.' });
      return;
    }

    // If Whisper already translated to English (only for small model), skip external translation
    let final;
    if (whisperWillTranslate) {
      log('Whisper already translated to English — skipping translation');
      final = cleanSegments(segments);
    } else if (!skipTranslation && translatorConfig) {
      const cleanedTrans = cleanSegments(segments);
      final = await translate(cleanedTrans, translatorConfig, device);
      // Clean again to catch translation hallucination loops
      final = cleanSegments(final);
    } else {
      final = cleanSegments(segments);
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
    log(`ERROR: ${err?.message ?? err}`);
    post('error', { message: err?.message ?? String(err) });
  }
};
