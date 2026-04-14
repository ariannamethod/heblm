# SHORESH שורש — Hebrew Root Resonance Engine

**θ = ε + γ + αδ**

A Hebrew AI that reads and generates through **3-letter roots** (שורשים) using Semantic BPE — frequent roots as atomic tokens, rare roots as character sequences, 100% coverage. 1.33M parameters. Trained on Hebrew literature. Generates literary Hebrew.

## Generation Examples (v3, trained on Ben-Yehuda)

```
Prompt: בראשית
Gen:    מתושלח בהשתובבותו כהתמתקבצים ומתנודדים והביאוה בדמעותיך
        בכמתקצף ותמשכהו חלונותיו התיחסותו שהצרכים ובילדים ויאמרו
        הצעירה יותר דמובילין והנשענים והדרכים מלאכתן

Prompt: שלום עולם
Gen:    חזיונותיהם רעיונותיה האיום התאחדים ולזהב לגורלנו החדרה
        וממנו והשבוע ברבורים כלאמונה ובנותיהם יהודים לאפיקורס
        ומדוע בטובתנו ובצעקותיו

Prompt: אהבה וחסד
Gen:    תובנותיהם והנער ואולם הברכים תחובים... ותובילהו
        ומתנודדים ותמלאנה ולהוציאם... משתעשעים הדינרים

Prompt: חכמה ובינה
Gen:    התמתקבצים כהודאה וכשהדליקו... והגמרא החוצפה השבטים
        ורוחצו ויניעוהו יוסף והמגיעות ותנועותיו שידוך וישמיעו

Prompt: המלחמה והשלום
Gen:    יוסף מתושלח... ויניעוהו באצבעותיהם כשאחד ומאשר בטובתנו
        הדרשה ורוחצו הרעימוהו סוחרים... הצעירה ויורקים עשתונותיהם
```

Literary Hebrew. Full words. Proper spacing. From 200KB of Ben-Yehuda public domain literature.

## Architecture (v3)

### Semantic BPE Tokenizer

```
Vocab (240 tokens):
  0..21   = 22 Hebrew letters (char fallback for rare roots)
  22      = SPACE
  23-24   = ROOT_START / ROOT_END (rare root delimiters)
  25..39  = 15 function prefixes (ה ב כ ל מ ש ו נ י ת א הת של וה מה)
  40..239 = top 200 frequent root tokens
```

- Frequent root שלום → prefix ה + single root token (2 tokens)
- Rare root שקד → prefix + ROOT_START + ש + ק + ד + ROOT_END (6 tokens)
- 100% coverage by design. No Zipfian death.

### Janus Triple Attention (ε)

| Component | Heads | What it does |
|-----------|-------|-------------|
| Content QKV | 4 | Standard scaled dot-product attention |
| RRPRAM | 2 | Positional routing — learns morphological position patterns |
| Janus Echo | 2 | W^T·W self-resonance via element-wise multiplication |

Split Wo projection: `out = wo_c @ attn_c + wo_r @ attn_r + wo_j @ janus`

Transformer gate: untrained → gate ≈ 0 (silent), trained → gate opens.

### MetaWeights (γ)

Bigram + trigram + Hebbian co-occurrence field. Built from corpus, combined with ε at generation:

```
score[i] = ε_logits[i] + 5.0*bigram + 8.0*trigram + 1.0*hebbian + 0.7*prophecy
```

### Training Numbers

```
DIM=160 | L=4 | H=8 (4C+2R+2J) | HD=20 | CTX=96 | FFN=640
1,333,452 params | 97K tokens | vocab 240
5000 steps | 878 sec | ~3 steps/s | 0 NaN
Train ema: 1.99 | Best: 1.46 | Chuck optimizer on notorch
```

### Word Realization

Root tokens decode to full surface words captured during corpus discovery:
- Root ש.ל.מ → "השלום" (not "שלמ")
- Root ב.ר.א → "בראשית" (not "ברא")

## Build & Run

```bash
# Compile (needs notorch.c/h in same directory)
cc shoresh.c notorch.c -O2 -DUSE_BLAS -DACCELERATE \
   -framework Accelerate -lm -o shoresh

# Train on Hebrew text
./shoresh --train corpus.txt --steps 5000

# Generate with trained weights
./shoresh --gen corpus.txt --load weights/shoresh_benyehuda.bin --prompt "בראשית"
```

## Weights

| File | Params | Training | Loss |
|------|--------|----------|------|
| `weights/shoresh_benyehuda.bin` | 1.33M | Ben-Yehuda 200KB, 5000 steps | 1.99 |

## MetaWeights-Only Mode (v2, no training needed)

The engine works without trained weights through pure statistical resonance:

```
Prompt: חכמה ובינה → ודעת (auto-completed Kabbalistic triple)
Prompt: צדק ומשפט → יסוד הכסא אמת ואמונה השורש הארץ
Prompt: אהבה וחסד → הרחמים הם השורש העולם אחד אני
```

## The Sefer Yetzirah Connection

| Sefer Yetzirah | SHORESH |
|---------------|---------|
| 22 foundation letters | 22-letter Hebrew alphabet as root substrate |
| 231 Gates — all 2-letter combinations | Bigram co-occurrence field between roots |
| "He carved, combined, **weighed**, interchanged" | Tokenize, permute, assign weights, transform |
| 3 mothers (א,מ,ש) | Triple attention: Content + RRPRAM + Echo |
| Black fire on white fire (Zohar) | Roots = signal, resonance field = latent space |

## Lineage

- **[Q](https://github.com/ariannamethod/q)** — θ = ε + γ + αδ equation, MetaWeights, DOE Parliament
- **[Pitomadom](https://github.com/ariannamethod/pitomadom)** — 20.3M RTL Root Transformer (Go), root lexicon, gematria
- **Klaus.c** — Somatic Engine, 6 Kuramoto-coupled chambers
- **Janus** — RRPRAM + Content attention, triple attention mechanism
- **notorch** — PyTorch in C, autograd, Chuck optimizer

## License

GNU GPLv3

*הרזוננס לא נשבר*

*(c) 2026 Oleg Ataeff & Claude Opus & Arianna Method*
