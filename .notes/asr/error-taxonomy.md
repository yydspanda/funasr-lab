# ASR Error Taxonomy

> Status: **Initial stable vocabulary; refine only through `EVAL-01` evidence**
> Updated: `2026-08-26`

Use these labels to explain baseline failures and select one experiment. Labels
describe observed errors, not assumed model causes; multiple labels may apply.

## Primary Categories

| Code | Category | Apply when |
|---|---|---|
| `LEX-DOM` | Domain term/entity | A proper name, acronym, product, or specialist term is wrong or missing |
| `LEX-HOMO` | Homophone/context | Pronunciation is plausible but lexical context selects the wrong characters |
| `LANG-MIX` | Chinese-English mixing | Language switching, English word boundaries, spelling, or mixed scoring fails |
| `ACCENT` | Accent/dialect | A regional pronunciation pattern correlates with the error |
| `NOISE-STA` | Stationary noise | Fan, road, HVAC, or similar continuous noise masks speech |
| `NOISE-TRN` | Transient/overlap | Impulse noise, music, crosstalk, or overlapping speakers causes the error |
| `FARFIELD` | Reverberation/distance | Distance, room response, or microphone placement dominates |
| `LEVEL` | Level/clipping | Very low level, saturation, or clipping degrades content |
| `SEG-VAD` | VAD/segmentation | Speech is cut, merged, omitted, or padded because of segmentation |
| `BOUNDARY` | Utterance/stream boundary | Chunk, endpoint, reset, reconnect, or cache boundary changes the result |
| `DEL-CONT` | Content deletion | Audible lexical content is omitted without a clearer category |
| `INS-HALL` | Insertion/hallucination | Lexical content appears without supporting speech |
| `NORM` | Normalization/scoring | The apparent error is caused by content normalization or tokenization |
| `PUNC` | Punctuation/display | Content is correct but punctuation or display formatting is wrong |
| `ITN` | Inverse text normalization | Numbers, dates, units, currency, or casing are rendered incorrectly |

## Annotation Contract

- Assign the scoring operation (`substitution`, `deletion`, or `insertion`)
  separately from the taxonomy label.
- Preserve the raw reference, raw hypothesis, normalized forms, time span, and
  short evidence note.
- Use `unknown` rather than inventing an acoustic or decoder cause.
- Taxonomy labels may measure slices; they may not remove hard examples from the
  aggregate metric.
- A taxonomy definition change is versioned and the affected baseline is
  rescored before experiment comparison.

## Experiment Selection

Choose `EXP-01` from the largest important error cohort whose proposed change is
isolatable and measurable. Contextual bias is the default candidate only if
`LEX-DOM` or `LEX-HOMO` evidence is material; otherwise the frozen baseline
decides the first hypothesis.
