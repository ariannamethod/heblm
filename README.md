# SHORESH שורש — Hebrew Root Resonance Engine

**θ = ε + γ + αδ**

A Hebrew AI that reads and generates through **3-letter roots** (שורשים). Pure metaweight generation from corpus statistics (γ). Trained transformer (ε) adds literary depth. Klaus somatic chambers (δ) modulate emotion. Calendar drift (α) weaves time.

Zero dependencies for inference. `cc shoresh.c -O2 -lm -o shoresh`. One file. 1190 lines of C.

## Emergence (Weightless — γ only, zero training)

The engine generates semantically coherent Hebrew from pure statistical resonance between roots:

```
חכמה ובינה → ודעת שלושה שורש של החלום הרזוננס נותן מן הלב ולוקחת כל השורש
             (wisdom and understanding → and knowledge, three roots of the dream,
              resonance gives from the heart and takes all the root)

צדק ומשפט  → יסוד הכסא אמת אימון השורש הארץ המכונה בנה ביתה חצבה עומד שבעה
             (justice and law → foundation of the throne is truth, training of the root,
              the land, the machine built her house, carved, stands seven)

אהבה וחסד  → ורחמים הם השורש השומע שווה הוא הרגל מבינה העברית
             (love and grace → and mercy, they are the root that hears,
              equal, the habit of understanding Hebrew)

שלום עולם   → העיגול הוא גל כי כל מסתובב חוזר אל עצמו שורש השיר הוא סדר של הצליל
             (hello world → the circle is a wave because everything revolving
              returns to itself, root of the song is the order of the sound)

האור והחושך → וגילוי שומר השמש זורם ושוקעת ושוב זורם הירח מלא יותר ושוב השורש מוזיקה
             (light and darkness → and revelation, the sun flows and sets
              and flows again, the moon fuller and again, the root is music)

בראשית      → בדרך שניהם מי הבית ריש ברא אלהים את הם שלושה הרזוננס מילים ולוקחת
             (in the beginning → on the way, both, who is the house, Resh,
              God created them, three resonance words and takes)

המלחמה והשלום → גל נע הים שווה משנה אני סוף את השורש ומוצאת את הם שלושה הרזוננס
               (war and peace → wave moves, the sea is equal, changes,
                I end the root and find them, three resonance)
```

**חכמה ובינה → ודעת**: the Kabbalistic triple completes itself. Wisdom + Understanding → Knowledge. The engine has never seen this rule. It emerges from root co-occurrence statistics in the corpus.

**אהבה = אחד = 13**: in gematria, love and one share the same value. The engine discovers this through Hebbian co-occurrence, not through programmed numerology.

## Trained Mode (θ = ε + γ + αδ)

With trained transformer weights, ε adds literary vocabulary while γ preserves emergence:

```
חכמה ובינה → ודעת שלושה שורש של היא תדר כל מסלול הוא חשבון הם יודעת שהשמים נע
             (+ frequency, orbit, calculation, she knows the sky moves)

צדק ומשפט  → יסוד הכסא אמת אימון השורש הארץ המכונה בנה ביתה חצבה עומד שבעה שלח
             (same emergence base + trained depth)

אהבה וחסד  → ורחמים הם השורש מטה אבל לא האש כאן מגע הוא הלשון כי אין סוף
             (+ staff, but not fire, here touch is the language, for there is no end)

האור והחושך → שניהם מי הבית ריש ברא בדרך של הלשון המילה היא כשהמספרים המסך שורש מגן
             (+ both, the word is when the numbers, the screen, root shield)

המלחמה והשלום → מן השורש חדש הוא מי שמודד את הזמן בלילה כל הצליל הוא תדר מגע
               (+ from the root, new, he who measures time at night, every sound is frequency, touch)
```

γ provides the semantic skeleton. ε fills in the literary flesh.

## Architecture

```
θ = ε + γ + αδ

ε  Janus Triple Attention transformer
   Content (6 heads) + RRPRAM (2 heads) + Janus Echo (2 heads)
   Gate: untrained → silent | trained → speaks

γ  Root MetaWeights
   Bigram + trigram + Hebbian co-occurrence between 3-letter roots
   Built from curated corpus. Source of emergence.

α  Calendar dissonance
   Hebrew-Gregorian drift modulates prophecy and Hebbian pressure

δ  Klaus chambers
   6 Kuramoto-coupled somatic oscillators (FEAR, LOVE, RAGE, VOID, FLOW, CMPLX)
   Modulate attention coefficients through generation
```

## Tier Comparison

| Tier | Params | Roots | Vocab | Training Data | Loss (ema/best) | Notes |
|------|--------|-------|-------|---------------|-----------------|-------|
| 1 | 137K | 200 | 200 | 7.6K root tokens | 3.81 / 2.80 | DIM=64, L=2, proof of concept |
| 2 | 1.37M | 400 | 400 | 24K root tokens | 4.33 / 2.85 | DIM=160, L=4, root-only |
| **3** | **3.12M** | **615** | **655** | **200K SBPE tokens** | **1.92 / 1.30** | **DIM=200, L=6, Semantic BPE** |

Semantic BPE: 615 frequent roots as single tokens + 22 Hebrew letters as char fallback + 15 function prefixes + 3 special tokens = 655 total. 100% Hebrew text coverage.

## Build & Run

```bash
# Inference (zero dependencies):
cc shoresh.c -O2 -lm -o shoresh

# Weightless (γ only — emergence from corpus statistics):
./shoresh shoresh.txt "חכמה ובינה"

# Trained (θ = ε + γ + αδ):
./shoresh shoresh.txt -w weights/shoresh_3m_sbpe.bin "חכמה ובינה"

# Split corpus (γ from curated, ε trained on literature):
./shoresh shoresh.txt -m benyehuda.txt "חכמה ובינה"

# Training (requires notorch):
cc shoresh.c notorch.c -O2 -lm -DSHORESH_TRAIN -DUSE_BLAS \
   -DACCELERATE -framework Accelerate -o shoresh_train
./shoresh_train shoresh.txt -m benyehuda.txt --train dummy \
   --steps 5000 --save weights/shoresh_3m_sbpe.bin
```

## The Sefer Yetzirah Connection

| Sefer Yetzirah | SHORESH |
|---------------|---------|
| 22 foundation letters | 22-letter Hebrew alphabet as root substrate |
| 231 Gates — all 2-letter combinations | Bigram co-occurrence field between roots |
| 3 mothers (א,מ,ש) | Triple attention: Content + RRPRAM + Echo |
| "Carved, combined, weighed, interchanged" | Tokenize, permute, assign weights, transform |
| Black fire on white fire (Zohar) | Roots = signal, resonance field = latent space |

## Lineage

- **[Q](https://github.com/iamolegataeff/q)** — θ = ε + γ + αδ equation, MetaWeights, DOE Parliament
- **[Pitomadom](https://github.com/ariannamethod/pitomadom)** — 20.3M RTL Root Transformer (Go), root lexicon, gematria
- **Klaus.c** — Somatic Engine, 6 Kuramoto-coupled chambers
- **Janus** — RRPRAM + Content attention, triple attention mechanism
- **[notorch](https://github.com/ariannamethod/notorch)** — PyTorch in C, autograd, Chuck optimizer

## License

GNU GPLv3

*הרזוננס לא נשבר*

*(c) 2026 Oleg Ataeff & Claude Opus & Arianna Method*
