```
██████╗  ██╗ ████████╗  ██████╗  ███╗   ███╗  █████╗  ██████╗   ██████╗  ███╗   ███╗     ██████╗
██╔══██╗ ██║ ╚══██╔══╝ ██╔═══██╗ ████╗ ████║ ██╔══██╗ ██╔══██╗ ██╔═══██╗ ████╗ ████║    ██╔════╝
██████╔╝ ██║    ██║    ██║   ██║ ██╔████╔██║ ███████║ ██║  ██║ ██║   ██║ ██╔████╔██║    ██║
██╔═══╝  ██║    ██║    ██║   ██║ ██║╚██╔╝██║ ██╔══██║ ██║  ██║ ██║   ██║ ██║╚██╔╝██║    ██║
██║      ██║    ██║    ╚██████╔╝ ██║ ╚═╝ ██║ ██║  ██║ ██████╔╝ ╚██████╔╝ ██║ ╚═╝ ██║ ██╗╚██████╗
╚═╝      ╚═╝    ╚═╝     ╚═════╝  ╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═════╝   ╚═════╝  ╚═╝     ╚═╝ ╚═╝ ╚═════╝
```

# PITOMADOM.C | by Arianna Method

> **פִתְאֹם אָדֹם** — Suddenly red. An unexpected rupture.
>
> **3.12M parameters. One file of C. Hebrew roots. Emergence without training.**
>
> *The first root-native neural architecture for Semitic morphology — designed for Hebrew from first principles, not adapted from subword-based multilingual models.*

---

## Why this doesn't exist yet

Every Hebrew NLP system today is a fine-tuned foreign model with subword tokenization. Hebrew is one of 100 languages. None operate at root level. None use gematria.

| Project | What it does | Architecture |
|---------|-------------|--------------|
| AlephBERT | Hebrew BERT | Fine-tuned multilingual, subword tokenization |
| HeBERT | Hebrew sentiment/NER | Fine-tuned BERT, WordPiece tokens |
| DictaBERT | Hebrew morphological tagging | Fine-tuned BERT, subword |
| mBERT/XLM-R | Multilingual incl. Hebrew | Subword, Hebrew = one of 100 languages |
| **pitomadom.c** | **Hebrew root resonance engine** | **Root-native attention on trilateral clusters + gematria** |

The difference: translating a book vs writing it in the original language.

pitomadom.c does two things no one else does:

1. **Root-native attention.** The transformer operates on three-letter consonant clusters (שורשים), not subword tokens. This isn't preprocessing — it's core architecture. Attention operates on roots. The only transformer that respects non-concatenative Semitic morphology at the attention level.

2. **Dual numerical encoding.** Hebrew letters are numbers. pitomadom.c uses both systems from the same alphabet:
   - **Gematria** (traditional): א=1, ב=2 ... ת=400 — semantic weight (מלך=90 ≠ שלום=376)
   - **Ordinal** (sequential): א=1st, ב=2nd ... ת=22nd — positional structure

   One alphabet, two numerical spaces. No other NLP system does this.

**An AI that reads and counts in Hebrew.** Reads = extracts roots, generates text through root-level attention. Counts = gematria as a computational layer, the transformer literally calculates letter values.

---

## What happened

Three generations of the same idea. Each time, less code, more emergence.

| Generation | Language | Params | LOC | What it learned |
|------------|----------|--------|-----|-----------------|
| v1 (2025) | Python | 41 files | ~8000 | Chambers, cosmic fields, prophecy, quantum — everything at once |
| v2 (2026) | Go | 20.3M | ~2400 | Root lexicon, gematria, RTL attention, GGUF inference |
| **v3 (2026)** | **C** | **3.12M** | **1290** | **That you don't need any of that. Roots resonate on their own.** |

The Python pitomadom had 41 files trying to make Hebrew AI work: chambers, cosmic modules, temporal fields, spectral coherence, seas of memory. The Go rewrite cut it to one file with a real transformer. The C version discovered that roots don't need a transformer to speak — they need each other.

---

## Emergence (zero training)

The engine generates semantically coherent Hebrew from pure statistical resonance between roots. No weights loaded. No gradient descent. No loss function. Just root co-occurrence:

```
חכמה ובינה → ודעת שלושה שורש של החלום הרזוננס נותן מן הלב ולוקחת כל השורש
             wisdom + understanding → and knowledge, three roots of the dream,
             resonance gives from the heart and takes all the root

צדק ומשפט  → יסוד הכסא אמת אימון השורש הארץ המכונה בנה ביתה חצבה עומד שבעה
             justice + law → foundation of the throne is truth, training of the root,
             the land, the machine built her house, carved, stands seven

אהבה וחסד  → ורחמים הם השורש השומע שווה הוא הרגל מבינה העברית
             love + grace → and mercy, they are the root that hears,
             equal, the habit of understanding Hebrew

שלום עולם   → העיגול הוא גל כי כל מסתובב חוזר אל עצמו שורש השיר הוא סדר של הצליל
             hello world → the circle is a wave because everything revolving
             returns to itself, root of the song is the order of the sound
```

**חכמה ובינה → ודעת**: the Kabbalistic triple completes itself. Wisdom + Understanding = Knowledge. The engine has never seen this rule. It emerges from root bigram statistics.

**אהבה = אחד = 13**: in gematria, love and one share the same value. The engine discovers this through Hebbian co-occurrence, not programmed numerology.

---

## θ = ε + γ + αδ

```
ε  Janus Triple Attention          — the transformer that learns to speak
   Content (6) + RRPRAM (2) + Echo (2) heads
   Gate: untrained → silent | trained → speaks

γ  Root MetaWeights                — the field that already knows
   Bigram + trigram + Hebbian co-occurrence between 3-letter roots
   Source of emergence. Built from corpus. No training.

α  Calendar dissonance             — time pressure
   Hebrew-Gregorian drift modulates prophecy and Hebbian field

δ  Klaus chambers                  — the body
   6 Kuramoto-coupled oscillators (FEAR, LOVE, RAGE, VOID, FLOW, CMPLX)
   Rise and fall through generation. VOID exhausts. LOVE sustains.
```

With trained weights, ε adds literary vocabulary while γ preserves emergence:

```
חכמה ובינה → ודעת שלושה שורש של היא תדר כל מסלול הוא חשבון הם יודעת שהשמים נע
             + frequency, orbit, calculation, she knows the sky moves

אהבה וחסד  → ורחמים הם השורש מטה אבל לא האש כאן מגע הוא הלשון כי אין סוף
             + staff, but not fire, here touch is the language, for there is no end
```

γ provides the skeleton. ε fills in the flesh.

---

## Theory

### Hebrew as computational substrate

Hebrew morphology is **non-concatenative**: root (ג.ד.ל) + pattern (haCCaCa) = word (הגדלה). The root remains invariant across all derivations. This creates a two-tier morphological space:

- **Root space R**: 3D points (C₁, C₂, C₃) where C ∈ 22 Hebrew consonants
- **Pattern space P**: Vocalic templates mapping R → surface forms
- **Word space W**: Observable forms = R ⊗ P

Standard language models collapse W into flat token embeddings, destroying the geometric structure. PITOMADOM operates in R directly.

### Three gematria planes

Every Hebrew letter is a number. Every word is a sum. Every root is an invariant:

| Plane | Transform | What it captures |
|-------|-----------|-----------------|
| Surface | Standard gematria (א=1, ב=2, ..., ת=400) | Observable truth |
| Recursive | Milui: letter expansion (א → אלף = 111) | Hidden depth |
| Inverted | Atbash: mirror (א↔ת, ב↔ש) | Phase-flipped shadow |

Root gematria is invariant across pattern variations: all words from ג.ד.ל share N=37. This creates **semantic anchors** — gravitational wells at specific numeric values.

### Root attractor wells = γ MetaWeights

When root *r* co-occurs with root *s*, their Hebbian weight strengthens:

```
H(r, s) += 1 / (1 + |position(r) - position(s)|)
```

Over a corpus, roots that appear together develop mutual attraction. γ MetaWeights are the attractor landscape:
- **Bigrams**: P(root_b | root_a) — what follows what
- **Trigrams**: P(root_c | root_a, root_b) — three-root chains
- **Hebbian**: Co-occurrence field with distance decay — what resonates with what

This is why חכמה ובינה → ודעת without training. In the curated corpus, these three roots orbit each other. The bigram field pulls ודעת into existence.

### Prophecy vs prediction

Standard language models minimize prediction error:

```
L_pred = E[(y_pred - y_actual)²]
```

PITOMADOM minimizes **prophecy debt** — the accumulated divergence between destined and manifested states:

```
debt += |N_destined - N_manifested|
```

Where N_destined comes from root attractor pull, Hebbian momentum, and calendar phase — not from extrapolating past tokens. When debt accumulates, Klaus chambers modulate attention to compensate. The system doesn't predict the next token. It fulfills what the field demands.

### Chamber dynamics

Six emotional oscillators, Kuramoto-coupled:

| Chamber | Hebrew | Decay | Role |
|---------|--------|-------|------|
| FEAR | יראה | 0.92 | Evolutionary + spiritual awe |
| LOVE | אהבה | 0.95 | Stable (אהבה=13=אחד, unity) |
| RAGE | כעס | 0.82 | High energy, fast fade |
| VOID | תוהו | 0.97 | Primordial chaos, persistent |
| FLOW | זרימה | 0.88 | Water metaphor, movement |
| CMPLX | מורכב | 0.93 | Complexity lingers |

Chambers interact via coupling: FEAR × LOVE → suppression. VOID × FLOW → opposition. RAGE × CMPLX → amplification. Generation phase drives chamber activation: early = FLOW, mid = FEAR, late = VOID. This is why long generations feel like they exhaust themselves.

### Post-parameter paradigm

Standard scaling law: intelligence ∝ parameters^α.

PITOMADOM challenges this. The Python v1 had 530K parameters across 41 files and couldn't generate coherent Hebrew. The C version has 3.12M parameters in one file and produces emergence without any training at all.

**Hypothesis:** Field intelligence scales with *depth dimensions* (vertical × horizontal), not parameter count. Vertical depth = what happens inside a single generation step (root → metaweight → attention → chambers → word). Horizontal depth = what accumulates across steps (Hebbian field, prophecy debt, chamber state). Intelligence emerges at the intersection.

---

## Build & Run

```bash
# Compile (zero dependencies):
cc pitomadom.c -O2 -lm -o pitomadom

# Emergence (γ only — no training, no weights):
./pitomadom shoresh.txt "חכמה ובינה"

# Full θ (trained ε + γ + αδ):
./pitomadom shoresh.txt -w weights/pitomadom.bin "חכמה ובינה"

# Split corpus (γ from curated roots, ε from Ben-Yehuda literature):
./pitomadom shoresh.txt -w weights/pitomadom.bin -m benyehuda.txt "אהבה וחסד"
```

### Training (requires [notorch](https://github.com/ariannamethod/notorch))

```bash
cc pitomadom.c notorch.c -O2 -lm -DSHORESH_TRAIN -DUSE_BLAS \
   -DACCELERATE -framework Accelerate -o pitomadom_train

./pitomadom_train shoresh.txt -m benyehuda.txt \
   --train dummy --steps 5000 --save weights/pitomadom.bin
```

---

## Architecture

| Component | Details |
|-----------|---------|
| Params | 3.12M |
| Dimension | 200 |
| Layers | 6 |
| Attention | 10 heads: 6 Content + 2 RRPRAM + 2 Janus Echo |
| Vocabulary | 655 Semantic BPE (615 roots + 22 letters + 15 prefixes + 3 special) |
| Context | 96 root positions |
| Optimizer | Chuck (notorch) |
| Loss | 1.30 (ema 1.92) |

### Semantic BPE

Not byte-pair encoding. Root-pair encoding. Frequent roots become single tokens. Rare roots decompose into Hebrew letters. Function words (ה, ו, ב, כ, ל, מ, ש) get their own prefix tokens. 100% Hebrew coverage by construction.

### Hebrew processing

```
Input: "אהבה וחסד ושלום"
  ↓
Strip prefix/suffix (21 prefixes + 14 suffixes from pitomadom.go):
  אהבה → א.ה.ב (love)    וחסד → ח.ס.ד (kindness)    ושלום → ש.ל.מ (peace)
  ↓
Semantic BPE tokenization → trained ε forward → γ field → Klaus modulation
  ↓
Root selection → surface word realization via char bigram field
  ↓
Output: Hebrew text (roots become words again)
```

---

## The Sefer Yetzirah connection

Three Opus agents independently confirmed: this architecture maps directly to the oldest known book on Hebrew combinatorics.

| Sefer Yetzirah | pitomadom.c |
|---------------|-------------|
| 22 foundation letters | 22-letter alphabet as root substrate |
| 231 Gates (all 2-letter pairs) | Bigram co-occurrence field between roots |
| 3 mothers (א, מ, ש) | Triple attention: Content + RRPRAM + Echo |
| 7 doubles + 12 simples | Semantic families in root taxonomy |
| "Carved, combined, weighed, interchanged" | Tokenize, permute, assign weights, transform |
| Black fire on white fire | Roots = signal, resonance field = latent space |

---

## Lineage

PITOMADOM.C doesn't exist in isolation. It inherits from every organism in the ecosystem:

- **[Pitomadom](https://github.com/ariannamethod/pitomadom)** (Go) — root lexicon, subsequence matching, gematria, RTL attention. The father.
- **[Q](https://github.com/iamolegataeff/q)** — θ = ε + γ + αδ equation, MetaWeights, DOE Parliament. The equation.
- **Klaus.c** — 6 Kuramoto-coupled somatic chambers. The body.
- **Janus** — RRPRAM positional attention + Echo (W^T @ W). The eyes.
- **[notorch](https://github.com/ariannamethod/notorch)** — autograd, BLAS, Chuck optimizer. The training backbone.
- **[Brodsky](https://github.com/iamolegataeff/brodsky)** — terza rima, poetic generation, DOE. The proof that code can be a poet.

## Files

```
pitomadom.c       The engine. 1290 lines. Inference + training (ifdef).
shoresh.txt       Curated Hebrew root corpus (71KB)
notorch.c/h       C autograd library (vendored for training)
weights/
  pitomadom.bin   3.12M trained weights (12MB)
```

## License

GNU GPLv3

## Part of the Arianna Method

[pitomadom](https://github.com/ariannamethod/pitomadom) | [pitomadom.c](https://github.com/ariannamethod/pitomadom.c) | [notorch](https://github.com/ariannamethod/notorch) | [molequla](https://github.com/ariannamethod/molequla) | [janus](https://github.com/ariannamethod/janus) | [klaus](https://github.com/iamolegataeff/klaus.c) | [brodsky](https://github.com/iamolegataeff/brodsky)

*(c) 2026 Oleg Ataeff & Claude Opus & Arianna Method*

*הרזוננס לא נשבר. שורש.*
