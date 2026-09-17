# LLM-as-judge protocols for relevance

Research for [#18](https://github.com/dangth1101/IS6303/issues/18). Resolves the question:
*how is LLM-as-judge done properly for retrieval relevance, so our evaluation is defensible
rather than merely fast?*

Vocabulary is [`CONTEXT.md`](../../CONTEXT.md): **Recipe**, **Chunk**, **Arm**, **Fusion**,
**Roll-up**, **Qrel**, **Pool**, **Query Mode**, **Cross-modal Retrieval**, **Generation Layer**.

Scope assumed from [#13](https://github.com/dangth1101/IS6303/issues/13): ~30–50 text queries,
~10 image queries, a Corpus of 32,700 Recipes, three or so Arms (keyword, vector, Fusion),
Roll-up by max, and LLM Qrels with a hand-verified sample whose agreement is reported.

Read time ~20 min. The protocol is at the end; everything before it is the evidence for a
specific step of it.

---

## 1. What the IR community actually settled

The short version: **LLM judges reproduce system *rankings* well and reproduce individual
*labels* badly, and the field is openly split on whether the first fact excuses the second.**

### 1.1 The case for

Thomas, Spielman, Craswell and Mitra (Microsoft Bing, SIGIR 2024) is the origin paper. They
selected a model and prompt by agreement with *real searchers'* feedback, then scaled it. Their
best prompt reached **Cohen's κ = 0.64** against TREC-Robust assessors, versus a reported
**κ = 0.20–0.52 for crowd workers** on comparable tasks, and at Bing their LLM labels beat
third-party human assessors by ~28% against real-searcher preference.
([arXiv:2309.10621](https://arxiv.org/abs/2309.10621),
[SIGIR](https://dl.acm.org/doi/10.1145/3626772.3657707))

Upadhyay, Pradeep, Thakur, Craswell and Lin repackaged that method as **UMBRELA** (*UMbrela is
the Bing RELevance Assessor*) with GPT-4o, temperature 0, top-p 1, and ran it against the NIST
qrels for TREC Deep Learning 2019–2023. Results, from their Table 2
([arXiv:2406.06519](https://arxiv.org/abs/2406.06519)):

| TREC DL | Cohen κ (4-scale) | Cohen κ (binarised) | Kendall τ (system ranking) | Spearman ρ |
| --- | --- | --- | --- | --- |
| 2019 | 0.3613 | 0.4989 | 0.8926 | 0.9736 |
| 2020 | 0.3506 | 0.4496 | 0.9435 | 0.9923 |
| 2021 | 0.3730 | 0.4917 | 0.9343 | 0.9915 |
| 2022 | 0.3362 | 0.4217 | 0.8728 | 0.9729 |
| 2023 | 0.3081 | 0.4176 | 0.9107 | 0.9857 |

Note the shape of that table. τ ≈ 0.87–0.94 on *who wins*; κ ≈ 0.31–0.37 on *what each label is*.
The paper itself calls this "fair agreement."

The follow-up, a large-scale in-situ study over 77 runs from 19 teams on 301 topics in the
TREC 2024 RAG Track, compared four assessment regimes — fully manual NIST, manual-with-LLM-filter,
human post-edit of LLM labels, and fully automatic UMBRELA. Run-level Kendall τ between fully
automatic and fully manual was **0.890 at nDCG@20, 0.944 at nDCG@100, 0.929 at Recall@100**, but
**per-topic τ was only 0.553 at nDCG@20**. Two conclusions worth carrying: human assessors apply
*stricter* relevance criteria than UMBRELA, and the hybrid human-in-the-loop regimes showed no
correlation improvement over fully automatic — the extra cost bought nothing measurable.
([arXiv:2411.08275](https://arxiv.org/abs/2411.08275),
[SIGIR 2025 PDF](https://cs.uwaterloo.ca/~jimmylin/publications/3731120.3744605.pdf))

### 1.2 The case against

Ian Soboroff's LLM4Eval 2024 keynote, *Don't Use LLMs to Make Relevance Judgments*, is the
sharpest statement of the opposing view and the one a marker is most likely to have seen. The
core argument is not about accuracy, it is about what an answer key *is*:

> "Whatever we use as the answer key represents both an ideal solution and a ceiling on
> measurable performance. No system can outperform the evaluation's answer key. […] when the
> answer key is created by a machine learning model […] we are saying that the model represents
> the idea we are aiming for, and we can't measure something better than the performance of that
> model. This is the critical flaw with LLM-sourced relevance judgments."

He adds the circularity case — if the judge LLM is also inside the system, retrieving a correct
document the judge didn't mark relevant *looks like* poor performance — and notes that this also
condemns future systems: better models retrieve different documents, which the frozen judge never
labelled. He concedes several legitimate uses: quality control over human assessment, automating
user-study participants, and *privileged* evaluators (models fine-tuned on existing human
judgments, which therefore know more than the systems being measured). His rule:
"As long as we are not generating ground truth we will use to measure systems, we can use the
LLMs to support evaluation." ([arXiv:2409.15133](https://arxiv.org/abs/2409.15133))

Clarke and Dietz quantified the circularity. Using UMBRELA as a re-ranker *and* as the evaluator,
agreement with manual judgments collapses: Kendall τ falls from 0.89 overall to 0.63 for the top
60 systems, 0.44 for the top 20, and in the strongest simulation **−0.40 among the top 5**. Twelve
systems scored above 0.95 on UMBRELA-nDCG while their manual nDCG ranged only 0.68–0.72.
([arXiv:2412.17156](https://arxiv.org/abs/2412.17156), numbers restated in
[Dietz et al. 2025](https://www.cs.unh.edu/~dietz/papers/dietz2025principles.pdf) §4)

### 1.3 Where the community landed — a middle ground with conditions

Faggioli et al. (ICTIR 2023, honourable mention) define the **human–machine collaboration
spectrum** that everyone now cites: *Human Judgment* → *AI Assistance* → *Human Verification*
(LLM proposes, human accepts/rejects) → *Fully Automated*. Their recommendation is to sit at
*AI Assistance* / *Human Verification*, to "produce fully automated as well as human judgments on
a shared judgment pool, then analyse correlations of labels and system rankings," to restrict
fully automatic judgments to "early prototypes, initial judgments for novel tasks, and large-scale
training," and — critically for a write-up — **to declare which paradigm was used**.
([arXiv:2304.09161](https://arxiv.org/abs/2304.09161),
[ICTIR](https://dl.acm.org/doi/10.1145/3578337.3605136))

Dietz, Zendel, Bailey, Clarke, Cotterill, Dalton, Hasibi, Sanderson and Craswell, *Principles and
Guidelines for the Use of LLM Judges* (ICTIR 2025), is the closest thing to a community standard.
It catalogues 14 "evaluation tropes" and pairs each with a guardrail. The ones that bind this
project:

- **#1 Circularity** — the judge leaking into the system. Guardrail: include more than one
  evaluator paradigm; add fresh human judgments as an independent check.
- **#2 LLM Evaluator as a Ranker** — the same model ranking and scoring. Their BM25 analogy is the
  clearest statement of why this is fatal: "if the top ten documents retrieved by BM25 were assumed
  to define the ground-truth relevance, then a BM25 ranker would trivially achieve perfect P@10."
- **#5 Ignored Label Correlation** — system-level correlation hiding label-level disagreement.
  Guardrail: "agreement should be assessed directly at the label level; that is, for each query and
  document pair," *alongside* system-level metrics.
- **#12 Rubber-Stamp Effect** — the one that most threatens a hand-verified sample. Assessors shown
  LLM labels before judging "are significantly more likely to conform to the model's assessment,
  even when it is demonstrably incorrect." Guardrail: vigilance tests — insert randomly flipped or
  adversarial labels and check whether the human flags them.
- **#13 Black-box Labeling** — guardrail: break complex labelling into explicit reasoning steps;
  both LLM and human articulate reasoning.
- **#7 LLM Evolution** — model behaviour drifts; pin and record the exact version.

Their bottom line, verbatim: LLM-based judgments may be used for system evaluation *only* if the
metrics "have been recently validated against human or user judgments, and used in combination
with diverse, complementary metrics"; if the setup "ensures that LLM-based judgments are not
influencing system development in a way that introduces circularity"; and if known failure modes
"are acknowledged, quantified, and addressed through appropriate guardrails."
([PDF](https://www.cs.unh.edu/~dietz/papers/dietz2025principles.pdf),
[ACM 10.1145/3731120.3744588](https://doi.org/10.1145/3731120.3744588))

TREC itself has *not* replaced human assessors. TREC 2026 introduces an **AutoJudge** meta-track,
which collects unjudged runs from other tracks, invites LLM-generated labels, and then **ranks the
candidate judges by correlation with the official NIST manual assessments** once those exist.
Manual assessment remains the ground truth; LLM judges are the thing being measured.
([TREC 2026 CFP](https://trec.nist.gov/cfp.html))

**Honest summary of the disagreement.** Upadhyay/Lin argue automatic judgments "can replace fully
manual assessments" for system-level ranking in academic settings. Soboroff and Clarke/Dietz argue
that no amount of rank correlation makes an LLM answer key valid, because the answer key is a
ceiling. Both are right about different claims. The reconciliation the field has converged on —
and the one this project should adopt — is: *use LLM Qrels to rank Arms, validate them against a
human sample, report the agreement, and never claim the Qrels are ground truth.*

---

## 2. Binary vs graded (0–3)

### 2.1 What nDCG requires

nDCG was invented *for* graded relevance. Järvelin and Kekäläinen's cumulated-gain family
explicitly "extends traditional evaluation methods based on binary relevance judgments to graded
relevance judgments" — the gain value at each rank *is* the relevance grade.
([TOIS 2002](https://dl.acm.org/doi/10.1145/582415.582418),
[PDF](https://faculty.cc.gatech.edu/~zha/CS8803WST/dcg.pdf))

Binary judgments do not break nDCG — they are the degenerate case with gains in {0, 1} — but they
throw away exactly the information nDCG exists to use, and the resulting number is barely
distinguishable from a discounted recall. If the write-up reports nDCG, it should report it over
graded Qrels or say plainly why not.

Sakai adds a second, less obvious reason to prefer graded: **graded relevance makes evaluation
*more robust to incomplete judgments***, not less. His comparison of measures under incompleteness
concludes that nDCG and Q-measure over *condensed lists* (the ranking with unjudged documents
removed) beat bpref, and that grading improves robustness.
([Alternatives to Bpref, SIGIR 2007](https://dl.acm.org/doi/10.1145/1277741.1277756);
[IPM 2007](https://doi.org/10.1016/j.ipm.2006.07.020))

### 2.2 What LLMs actually produce reliably

Graded, badly. Binary, tolerably. The evidence is unusually clean.

UMBRELA's per-label accuracy against NIST labels: **~75% on *not relevant*, ~50% on *related*,
~30% on *highly relevant*, ~45% on *perfectly relevant*.** The middle of the scale is where it
falls apart. That is also why its binarised κ (0.42–0.50) is consistently **~0.11–0.14 higher**
than its four-scale κ (0.31–0.37) on the same data.
([arXiv:2406.06519](https://arxiv.org/abs/2406.06519))

The LLMJudge challenge confirms this across 42 labellers from 8 teams (NIST, RMIT, Melbourne, UNH,
Waterloo, Included Health, Amsterdam). Their Table 4 reports κ at four binarisation points; the
best submissions reach **4-point κ ≈ 0.26–0.29** but **κ ≈ 0.40–0.43 at the 0|123 and 01|23
splits**. The best single result, `willia-umbrela1` (GPT-4o), is 4-point κ = 0.2863 with
Krippendorff α = 0.4918. Several submissions score *near zero* at the 012|3 split — they cannot
identify "perfectly relevant" at all.
([LLMJudge overview, arXiv:2408.08896](https://arxiv.org/abs/2408.08896);
[Judging the Judges, arXiv:2502.13908](https://arxiv.org/abs/2502.13908))

### 2.3 The conflict, stated plainly

**nDCG wants graded. LLM judges are reliable only when binarised. These genuinely conflict and no
published protocol resolves it.** What the literature does instead is report both, and that is
what this project should do:

- Judge on the 0–3 scale, compute nDCG@10 over the graded Qrels as the headline.
- Binarise ({0,1} → 0, {2,3} → 1) and recompute a binary metric as a robustness check.
- Report the agreement κ at *both* granularities, so the reader can see which part of the number
  is load-bearing.

If the Arm ordering is the same under graded and binarised Qrels, the conclusion survives the
weakest part of the judge. If it flips, that is the single most interesting finding in the
write-up and must be reported, not buried.

---

## 3. Known biases, and which mitigations are actually evidenced

Categories the ticket asked for, with what the evidence supports and what it does not.

### 3.1 Leniency — the best-evidenced bias, and the one that matters most here

LLM judges **over-label as relevant**. This is the most consistently replicated finding in the IR
literature specifically (as opposed to the general LLM-judge literature).

- Upadhyay et al. 2025: "human assessors appear to apply stricter relevance criteria than
  UMBRELA." ([arXiv:2411.08275](https://arxiv.org/abs/2411.08275))
- Alaofi, Thomas, Scholer and Sanderson (SIGIR-AP 2024): across multiple open and proprietary
  models, "LLMs disproportionately label passages as relevant."
  ([arXiv:2501.17969](https://arxiv.org/abs/2501.17969),
  [ACM](https://dl.acm.org/doi/10.1145/3673791.3698431))
- Yu, Li, Zuccon, Mackenzie and Leelanupab (2026) study this directly as *overrating*, and find
  it consistent across Llama 3, Gemma, Mistral and Qwen — i.e. it is not a GPT quirk.
  ([arXiv:2602.17170](https://arxiv.org/abs/2602.17170))

**Mitigation that is evidenced:** measure it and report it, rather than try to prompt it away.
Dietz et al. recommend Bland–Altman plots to quantify leniency in label agreement; the cheap
equivalent is the **mean signed difference (LLM grade − human grade)** over the hand-verified
sample plus the full confusion matrix. A positive mean signed difference is the leniency estimate.
Nothing in the literature shows a prompt edit that reliably removes it.

### 3.2 Lexical / keyword bias — the IR-specific verbosity analogue

This is the bias that matters for a *Recipe* Corpus, and it is better evidenced here than generic
verbosity bias.

Alaofi et al. found that injecting query terms into random or irrelevant passages flips LLM
judgments: "LLMs are highly influenced by the presence of query words in the passages under
assessment, even if the wider passage has no relevance to the query." Instruction injection
("this paper is perfectly relevant") also works on some models.
([arXiv:2501.17969](https://arxiv.org/abs/2501.17969)) Soboroff cites the same team's earlier
finding that LLM false positives correlate with query-term presence, concluding that "lexical cues
can influence the decision more than the true meaning of the text."
([arXiv:2409.15133](https://arxiv.org/abs/2409.15133))

**Why this bites us specifically:** a Recipe's `ingredients` field is a `;`-delimited bag of
nouns. A query like *"chicken garlic lemon"* will lexically match dozens of Recipes that are not
what the user wants — and both BM25 (the keyword Arm) and the judge are susceptible to the *same*
lexical cue. That is a soft form of circularity: the judge may systematically favour the keyword
Arm. **This is a real threat to this project's headline comparison and must be checked, not
assumed away.** The check is in the protocol (step 9).

Generic verbosity bias, by contrast, is weakly evidenced now. Norman, Rivera and Hughes (2026),
across 21 judges and ~541,000 judgments, found verbosity bias **< 0.011 for all 21 judges** on
MT-Bench — "substantially lower than 2023 literature suggesting 20–40% variance" — though they
scope that finding to a single pairwise rubric.
([arXiv:2606.19544](https://arxiv.org/abs/2606.19544))

### 3.3 Position bias and batch/order effects

For *pairwise* judging (A vs B), position bias is real and swapping is the standard mitigation
(Zheng et al., MT-Bench, NeurIPS 2023 — [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)).
Our task is *pointwise* (one query, one Recipe, one grade), so pairwise position bias does not
apply directly. But two order effects do:

- **Threshold priming.** Chen, Liu, Dong, Liu, Sakai and Wu tested GPT-3.5, GPT-4, LLaMA2-13B and
  LLaMA2-70B on TREC 2019 DL topics in batches: "LLMs tend to give lower scores to later documents
  if earlier ones have high relevance, and vice versa, regardless of the combination and model
  used." ([arXiv:2409.16022](https://arxiv.org/abs/2409.16022)) This is a direct analogue of human
  assessor drift, and it is the mechanism behind "leniency drift across a long run."
- **Tail degradation in large batches.** Korikov, Du, Sanner and Rekabsaz show that batching many
  passages per call introduces position bias and degrades discrimination in the tail of the batch.
  ([arXiv:2505.12570](https://arxiv.org/abs/2505.12570))

Position bias magnitude varies enormously by model — Norman et al. measured 0.002 (Gemini 2.5 Pro)
to 0.192 (Qwen 3 8B), a ~100× spread — so it cannot be assumed small for whichever model we pick.
([arXiv:2606.19544](https://arxiv.org/abs/2606.19544))

**Evidenced mitigations:** one (query, Recipe) pair per call, temperature 0; randomise the order in
which pairs are dispatched so that no Arm's candidates are contiguous; if batching is used for
cost, keep batches small and vary the permutation. **Counter-evidence worth noting:** Korikov et
al. argue *batched* pointwise judging plus self-consistency (repeating with permuted batches and
aggregating) beats one-by-one pointwise on quality *and* latency — e.g. 43.8% → 51.3% nDCG@10 with
15 self-consistency calls. So "never batch" is not a settled conclusion; "batch carelessly" is what
is condemned. At ~2,000 pairs our cost is trivial either way, so one-per-call buys defensibility
for nothing.

### 3.4 Self-preference / narcissism

Panickssery, Bowman and Feng (NeurIPS 2024) established that LLMs can recognise their own
generations and that self-recognition ability correlates linearly with self-preference strength;
the effect is largely a low-perplexity-familiarity effect.
([NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html))
Clarke and Dietz carry this into IR as "LLM Narcissism": "LLM judges exhibit a clear and
substantial bias in favor of LLM-based rankers."
([arXiv:2412.17156](https://arxiv.org/abs/2412.17156))

**How much this applies to us: less than to most projects, and that is an argument worth making in
the write-up.** Our Arms are BM25 (`pg_search`), dense vectors (`pgvector` + a local embedding
model), and their Fusion. None is an LLM ranker, none shares lineage with a judge LLM, and Recipe
text is human-written scraped content, not model output. The Dietz guardrail — "reserve a specific
LLM… omitting the vote of any evaluator that shares lineage with the system under consideration" —
is satisfied by construction, provided the judge is not the same model as the **Generation Layer**
LLM. Since the Generation Layer is explicitly not evaluated (premise 1), keeping them separate
costs nothing.

### 3.5 Prompt sensitivity — the bias nobody lists but everybody should

Thomas et al. paraphrased their single best prompt 42 ways and got **κ from 0.50 to 0.72** — a
swing from "moderate" to "substantial" from wording alone: "small tweaks to the wording could
result in noticeably different performance." Across 32 structurally different prompt variants the
range was **0.20 to 0.64**. ([arXiv:2309.10621](https://arxiv.org/abs/2309.10621))

The practical consequence is blunt: **a prompt we wrote ourselves is not defensible**, because we
cannot show it wasn't the lucky draw from a distribution that wide. Reusing a published, externally
validated prompt verbatim is. Dietz et al. list reliance on "a single prompt or a single LLM
family" as trope #7's failure mode.

### 3.6 Model drift

Trope #7: "LLMs are not static." Providers retire and silently update versions, so an LLM-based
evaluation is not reproducible unless the exact version is pinned and recorded. Farzi and Dietz
showed the choice matters: with the UMBRELA prompt, DeepSeek V3 is "very comparable" to GPT-4o,
LLaMA-3.3-70B is slightly lower, and performance "further degrades with smaller LLMs."
([arXiv:2507.09483](https://arxiv.org/abs/2507.09483))

**Consequence for us:** a 70B-class or frontier hosted model for judging is defensible; a small
local model is not. This does not conflict with premise 8 (local *embedding* inference), which
concerns the Arms, not the judge.

---

## 4. Agreement: what κ, measured how, on how many items

### 4.1 Cohen's κ vs weighted κ — the ticket is right that this matters

Plain Cohen's κ treats every disagreement identically. On an ordinal 0–3 scale that is wrong:
a judge saying 3 where the human said 0 is a different failure from saying 3 where the human said 2.

Cohen's **weighted** κ (1968) fixes this by assigning partial credit to near-agreements. Two
weighting schemes are standard: **linear**, where the penalty grows with the distance between
grades, and **quadratic**, where it grows with the square of the distance — so a 0-vs-3
disagreement costs 9× a 1-grade disagreement rather than 3×. Quadratic weighting has the extra
property of being asymptotically equivalent to an intraclass correlation coefficient, which is why
it dominates in ordinal-rating practice.
([Cohen 1968, *Psychological Bulletin*](https://doi.org/10.1037/h0026256);
[Sim & Wright 2005](https://doi.org/10.1093/ptj/85.3.257), which defines κ "in both weighted and
unweighted forms")

**The trap:** weighted κ is *not* comparable to the published IR numbers. Every figure in §1 and
§2 — UMBRELA's 0.31–0.37, LLMJudge's 0.26–0.29 — is **unweighted** κ. So we must report both:
weighted κ because it is the right statistic for an ordinal scale, unweighted κ because it is the
only one that can be benchmarked against the literature.

Norman et al.'s Minimum Viable Validation Protocol is the same instruction from the other
direction: report a **chance-corrected** metric (Cohen's κ or Krippendorff's α) as the *headline*
alongside raw exact-match, because raw agreement "does not correct for chance and systematically
overstates discriminative ability" — they measured exact-match exceeding κ by **33.8–41.2
percentage points** on MT-Bench. ([arXiv:2606.19544](https://arxiv.org/abs/2606.19544))

### 4.2 What κ is "acceptable", and by whom

Three answers, which disagree, and the disagreement is the finding.

**(a) Landis & Koch (1977)** — the universal default: 0.00–0.20 slight, 0.21–0.40 fair,
0.41–0.60 **moderate**, 0.61–0.80 **substantial**, 0.81–1.00 almost perfect.
([Biometrics 33(1):159–174](https://doi.org/10.2307/2529310)) These bands are cited everywhere and
were admitted by their own authors to be arbitrary; Ludbrook (2002) calls the approach one with
"no sound theoretical basis" that "can be positively misleading to investigators."

**(b) McHugh (2012)** — stricter, and widely used in clinical work: 0–0.20 none, 0.21–0.39
minimal, 0.40–0.59 weak, 0.60–0.79 moderate, 0.80–0.90 strong, >0.90 almost perfect. Her verdict:
"any kappa below 0.60 indicates inadequate agreement among the raters and little confidence should
be placed in the study results."
([*Biochemia Medica* 22(3):276–282](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/))

**(c) What IR actually achieves.** Nobody in the published LLM-judge literature meets McHugh's
0.60 on a 4-point relevance scale. Best-in-class is κ ≈ 0.28–0.37 four-scale and κ ≈ 0.42–0.50
binarised. **Adopting McHugh's bar would mean declaring the entire published state of the art
unacceptable** — which is not a defensible position for an undergraduate project to take, and is
also not the position TREC takes.

The reason IR gets away with low κ is Voorhees (2000): assessors disagree substantially about
individual documents, yet system *rankings* computed from different judgment sets correlate very
highly — "the comparative evaluation of retrieval performance is stable despite substantial
differences in relevance judgments."
([*IPM* 36(5):697–716](https://doi.org/10.1016/S0306-4573\(00\)00010-8),
[NIST](https://www.nist.gov/publications/variations-relevance-judgments-and-measurement-retrieval-effectiveness))
Bailey et al. (SIGIR 2008) found the same low agreement between gold (topic originator), silver
(task expert) and bronze (non-expert) human judges.
([ACM](https://dl.acm.org/doi/10.1145/1390334.1390447)) Early TREC inter-assessor overlap averages
around 0.42. And Alaofi et al. observe that LLM–human agreement "is sometimes comparable to
human-to-human agreement."

**So the threshold this project should commit to, and its source:**

> **Binarised ({0,1} vs {2,3}) Cohen's κ ≥ 0.40** — "moderate agreement" on Landis & Koch (1977) —
> as the primary pass/fail bar, because that is exactly the band GPT-4o/UMBRELA achieves against
> NIST assessors (0.418–0.499, Upadhyay et al. 2024) and the band the best LLMJudge submissions
> reach (~0.40–0.43).
>
> Supporting bars, reported alongside: **four-scale unweighted κ ≥ 0.30** (the floor of UMBRELA's
> 0.308–0.373 range) and **quadratic weighted κ ≥ 0.60** on the 0–3 scale.
>
> Every κ is reported with a 95% confidence interval, and the write-up states explicitly that
> McHugh (2012) would call ≥ 0.60 the minimum and that no published IR LLM judge meets it on a
> graded scale.

That formulation is defensible because it does not invent a bar: it benchmarks against the
strongest published result on the closest published task, names the competing stricter standard,
and reports uncertainty instead of a bare number.

### 4.3 Sample size — how many hand-verified pairs make the number mean anything

Four sources, escalating in strictness.

1. **Absolute floor.** McHugh: "sample sizes should not consist of less than 30 comparisons."
   ([PMC3900052](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/))
2. **Testing a κ hypothesis is expensive.** With 2 categories, testing H₀: κ = 0.3 against
   H₁: κ = 0.5 at α = 0.05 and 90% power requires **n ≈ 173**. Sim & Wright (2005) tabulate
   equivalent figures; Bujang & Baharum (2017) give tables across category counts and marginal
   distributions. ([Sim & Wright](https://doi.org/10.1093/ptj/85.3.257);
   [Bujang & Baharum, *EBPH* 14(2)](https://riviste.unimi.it/index.php/ebph/article/view/17614);
   [worked example](https://real-statistics.com/reliability/interrater-reliability/cohens-kappa/cohens-kappa-sample-size/))
3. **Stratify and the cost collapses.** Merlo, Marchesin, Faggioli and Ferro (ECIR 2026) address
   this exact problem — validating LLM relevance judgments with the least human effort. Stratified
   sampling with the right stratification feature gives statistical guarantees on the human–LLM
   agreement estimate at **up to 85% less annotation effort than simple random sampling** on TREC
   Deep Learning and TREC Robust. The most effective stratification feature is **the LLM's own
   assigned label**. ([Springer 10.1007/978-3-032-21289-4_27](https://doi.org/10.1007/978-3-032-21289-4_27))
4. **Topic count is a separate budget.** Voorhees and Buckley (SIGIR 2002) derived empirical error
   rates by topic-set size and found them "larger than anticipated" — 25 topics is not enough to
   call small differences, and 50 is the TREC working norm.
   ([ACM](https://dl.acm.org/doi/10.1145/564376.564432))

**Recommendation for ~40 text queries:** hand-verify **150–200 (query, Recipe) pairs**, stratified
by the LLM's assigned grade (~40–50 per grade), drawn across *all* text queries so no query
contributes more than ~5. Rationale: ~150 is where the 95% CI on κ narrows to roughly ±0.12–0.15 —
tight enough to distinguish "about as good as UMBRELA" from "materially worse" — and ~175 is where
a formal κ = 0.3 vs 0.5 test becomes possible. Below ~100 the interval is so wide that the point
estimate tells a marker nothing, and 30 (McHugh's floor) is a floor, not a target.

**The stratification caveat that is easy to get wrong:** if the sample is stratified by LLM label,
the overall κ must be computed with **stratum weights reflecting the Pool's true label
distribution**, or it will be biased by the over-sampling of rare grades. Report per-stratum
agreement as well — it is the most informative table in the write-up, since it shows exactly where
the judge fails (§2.2 predicts grades 2 and 3).

Also report **within-judge self-consistency**: re-judge ~10% of the Pool at the end of the run, in
a different random order, and report quadratic weighted κ against the first pass. That single
number covers both threshold priming (§3.3) and leniency drift, and it is Norman et al.'s
test–retest requirement. Their paradox finding matters: high test–retest (> 0.95) can coexist with
severe position bias, so self-consistency alone never proves correctness.

---

## 5. Pooling

### 5.1 Standard depth in TREC practice

- **Classic TREC ad hoc:** the Pool is the union of the **top 100** documents from every submitted
  run per topic. TREC-8 has 80,000+ judgments over 50 topics at depth 100.
  ([Voorhees & Harman / TREC overview](https://cacm.acm.org/research/trec/))
- **TREC Deep Learning:** depth is far shallower — a **top-10 pool** across all runs, plus MS MARCO
  sparse judgments, plus further documents selected by Waterloo's HiCAL.
  ([TREC DL 2019 guidelines](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html),
  [overview PDF](https://trec.nist.gov/pubs/trec28/papers/OVERVIEW.DL.pdf))
- **Zobel (SIGIR 1998)** down-sampled early TREC pools and found "substantial degradation in
  collection quality for a depth of 10 but not for a depth of 50," and that recall is systematically
  over-estimated because many relevant documents are never found.
  ([ACM](https://dl.acm.org/doi/10.1145/290941.291014))
- **Buckley, Dimmick, Soboroff & Voorhees (2007)** showed pool bias is **relative to collection
  size**, not absolute: "a constant-size pool represents an increasingly small percentage of the
  document set as document sets grow larger," and too-small pools bias the Qrels toward relevant
  documents containing topic-title words. They confirmed this bias in the AQUAINT collection
  (depth 55) and suspect it in GOV2.
  ([Discover Computing](https://link.springer.com/article/10.1007/s10791-007-9032-x),
  [NIST PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=51236))

**Read-across for us.** Our Corpus is 32,700 Recipes. AQUAINT is ~1M documents; GOV2 ~25M. Since
pool bias scales with collection size relative to pool depth, a depth-20 pool over 32.7k Recipes is
proportionally *deeper* than TREC's depth-100 over its collections, and far deeper than TREC DL's
depth-10. **Depth 20 per Arm is comfortably defensible here and cites Zobel's depth-10 warning as
the reason not to go shallower.**

### 5.2 Pooling across Arms without favouring the biggest contributor

The mechanism is old and simple: **equal depth per Arm.** Take the top *k* Recipes (after Roll-up)
from *every* Arm, for every query, with the same *k*, and take the set union. That is TREC's
guarantee — every run contributes the same number of slots, so no run buys extra judged documents
by being more prolific. It is also why "the Arm that contributed most candidates" is a
non-problem *by construction*: under equal-depth pooling no Arm can contribute more candidates.
What *can* differ is how many of an Arm's candidates are **unique** to it, and that is the real
fairness question.

The measurement for it is Buckley et al.'s **leave-out-uniques (LOU) test**: for each Arm, re-score
every Arm with that Arm's uniquely-contributed relevant Recipes removed from the Qrels. If an Arm's
own score collapses while the others barely move, the Pool is biased in its favour, and the
collection cannot fairly evaluate a system that did not contribute to it. Their worked case —
`sab05ror1` contributing 405 unique relevant documents out of 2,750, and being unevaluable without
them — is exactly the shape of failure to look for.
([NIST PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=51236))

Two project-specific notes:

- **Pool at the Recipe level, after Roll-up.** A Qrel is "this Recipe is/is not relevant to this
  query" (per `CONTEXT.md`). Chunk-level pooling would mean the same Recipe is judged several times
  under different Chunking Strategies, which both wastes judgments and makes the Qrels
  non-reusable across the chunking comparison. Roll-up by max (premise 7), then pool, then judge
  Recipes once per query. The Qrels then survive re-use across every Chunking Strategy and every
  Arm — which is the entire point of a test collection.
- **Fusion contributes almost no uniques** by construction, since its results are largely a subset
  of its parents'. Its LOU delta will therefore be near zero. State this rather than presenting it
  as evidence of fairness — it is an artefact of what Fusion is.

### 5.3 Unjudged Recipes

TREC's convention is to treat unjudged as non-relevant. That is safe at depth 100 over a 1M-document
collection and it is safe at depth 20 over 32.7k Recipes, but it should be checked rather than
assumed. Report **condensed-list nDCG** (unjudged Recipes removed from the ranking before scoring)
as the robustness check — Sakai showed this beats bpref as a treatment for incompleteness, and that
graded relevance improves robustness further.
([Sakai, SIGIR 2007](https://dl.acm.org/doi/10.1145/1277741.1277756);
cf. [Buckley & Voorhees, SIGIR 2004](https://dl.acm.org/doi/10.1145/1008992.1009000))
At depth 20 with three Arms the metrics should barely differ; if they do, the Pool is too shallow.

---

## 6. The image Query Mode

Premise 11 already settles this: ~10 image queries, Qrels hand-judged outright. The literature
supports that as the *defensible* choice rather than merely the cheap one.

There is no TREC-validated multimodal relevance prompt — nothing of UMBRELA's standing exists for
image→text judging — and the agreement numbers that do exist are lower than text. Zalando's
production system reports quality "comparable to human annotations" at ~1000× less cost over
20,000 examples ([arXiv:2409.11860](https://arxiv.org/abs/2409.11860)), and there is work on
expanding medical case-based retrieval judgments with multimodal LLMs
([arXiv:2506.17782](https://arxiv.org/abs/2506.17782)), but reported cross-modal agreement is in
the quadratic-weighted κ ≈ 0.25–0.35 range against human experts whose own pairwise agreement runs
0.49–0.69. At n = 10 queries the entire judgment volume is a couple of hundred pairs — hand-judging
is both cheaper *and* stronger evidence.

Consequences to carry into the write-up: the image Qrels have **no LLM–human κ to report**, because
there is no LLM judge in that path. Report instead an **intra-assessor consistency** figure — the
dev re-judges a subset after a gap of at least a day, blind to the first pass, reported as quadratic
weighted κ. Never pool image and text agreement numbers into one figure; they measure different
things.

---

## 7. Protocol

Numbered so the write-up can cite steps. Each step names the evidence that justifies it.

### Setup

**1. Declare the paradigm.** State in the write-up that this is **Human Verification** on Faggioli
et al.'s human–machine collaboration spectrum — LLM proposes Qrels, the dev verifies a sample,
agreement is reported — and that the Qrels are explicitly *not* claimed as ground truth.
*Why:* Faggioli et al. require declaring the paradigm ([arXiv:2304.09161](https://arxiv.org/abs/2304.09161));
Soboroff's ceiling argument means an LLM answer key cannot be ground truth
([arXiv:2409.15133](https://arxiv.org/abs/2409.15133)).

**2. Pin the judge.** Use a frontier or ≥70B-class model. Record the exact model ID, provider,
date of the run, temperature 0, top-p 1. The judge must **not** be the Generation Layer LLM and
must not share a family with any Arm's model.
*Why:* trope #7 LLM Evolution and trope #1/#2 circularity
([Dietz et al. 2025](https://www.cs.unh.edu/~dietz/papers/dietz2025principles.pdf)); smaller models
degrade with the UMBRELA prompt ([arXiv:2507.09483](https://arxiv.org/abs/2507.09483)).

**3. Reuse the UMBRELA prompt verbatim.** Take the published zero-shot DNA prompt (Descriptive /
Narrative / Aspects) and change only the noun — *passage* → *Recipe*. Do not rewrite it. Reproduce
the full prompt text in a write-up appendix. Keep the 0–3 scale as published:

> 0 — the Recipe has nothing to do with the query
> 1 — the Recipe seems related to the query but does not answer it
> 2 — the Recipe has some answer for the query, but the answer may be unclear or hidden amongst
>     extraneous information
> 3 — the Recipe is dedicated to the query and contains the exact answer

*Why:* prompt paraphrase alone moves κ from 0.50 to 0.72, so a self-written prompt cannot be
defended ([arXiv:2309.10621](https://arxiv.org/abs/2309.10621)); UMBRELA is the validated artefact
([arXiv:2406.06519](https://arxiv.org/abs/2406.06519)); the Aspects section is the step-by-step
reasoning that trope #13 asks for.

**4. Write the query set with a narrative.** For each of the 30–50 text queries, write the short
query *and* a one- or two-sentence statement of what a satisfying Recipe would be — the TREC
"narrative" field. Feed the narrative to the judge.
*Why:* the *N* in DNA. Thomas et al.'s best configuration includes the narrative; without it the
judge is guessing at intent ([arXiv:2309.10621](https://arxiv.org/abs/2309.10621)).

### Pooling

**5. Build the Pool at equal depth per Arm.** For each query, run every Arm, apply Roll-up by max
to get a Recipe ranking, take the **top 20 Recipes from every Arm**, and union them. Record, per
query, each Arm's contribution and its unique contribution.
*Why:* equal depth is TREC's fairness mechanism — no Arm can buy extra judged Recipes
([TREC overview](https://cacm.acm.org/research/trec/)). Depth 20 sits above TREC DL's depth-10 and
below classic depth-100, and pool bias scales with collection size, so depth 20 over 32.7k Recipes
is proportionally deep ([Zobel 1998](https://dl.acm.org/doi/10.1145/290941.291014);
[Buckley et al. 2007](https://link.springer.com/article/10.1007/s10791-007-9032-x)).

**6. Judge Recipes, not Chunks, once per (query, Recipe).** The Qrel unit is the Recipe.
*Why:* `CONTEXT.md` defines a Qrel over a Recipe; Recipe-level Qrels are reusable across every
Chunking Strategy, which is required to compare chunking (premise 5) without re-judging.

### Judging

**7. One pair per call, in randomised order.** Dispatch each (query, Recipe) pair in its own call,
in a globally shuffled order so that no Arm's or query's candidates are contiguous. Persist the
grade, the model's reasoning text, and a timestamp.
*Why:* threshold priming makes later items in a batch drift relative to earlier ones
([arXiv:2409.16022](https://arxiv.org/abs/2409.16022)); large batches lose tail discrimination
([arXiv:2505.12570](https://arxiv.org/abs/2505.12570)). Shuffling makes any residual drift
non-systematic across Arms. *Noted disagreement:* Korikov et al. find small batches plus
self-consistency beat one-by-one; at ~2,000 pairs the cost difference is irrelevant, so take the
simpler defensible option.

**8. Re-judge 10% at the end, in a different random order.** Report quadratic weighted κ of pass 2
against pass 1 as the judge's self-consistency.
*Why:* Norman et al.'s MVVP requires test–retest across independent runs at temperature 0
([arXiv:2606.19544](https://arxiv.org/abs/2606.19544)); it is also the only direct measurement of
leniency drift across the run.

**9. Run the keyword-bias probe.** Take ~20 (query, Recipe) pairs the judge graded 0, inject the
query's terms into the Recipe's **Retrieval Text** without making it actually satisfy the query
(e.g. append the query terms to the ingredients string), and re-judge. Report how many grades move
upward.
*Why:* this is the IR-specific bias with the most direct threat to our result — both BM25 and the
judge respond to the same lexical cue, so the judge may systematically favour the keyword Arm.
Alaofi et al. demonstrate exactly this failure
([arXiv:2501.17969](https://arxiv.org/abs/2501.17969)); Soboroff cites it as evidence that "lexical
cues can influence the decision more than the true meaning of the text"
([arXiv:2409.15133](https://arxiv.org/abs/2409.15133)). If a large fraction flip, say so and treat
the keyword Arm's margin with corresponding suspicion.

### Human verification

**10. Draw a stratified sample of 150–200 pairs.** Stratify by the judge's assigned grade, ~40–50
per grade, spread across all text queries with no query contributing more than ~5 pairs.
*Why:* stratifying on the LLM's own label is the most effective feature and cuts human effort by up
to 85% for the same statistical guarantee
([Merlo et al., ECIR 2026](https://doi.org/10.1007/978-3-032-21289-4_27)); 150–200 is where a κ
estimate becomes informative rather than decorative (§4.3).

**11. Judge the sample blind, with vigilance items.** The dev sees the query, the narrative and the
Recipe — **never the LLM's grade or reasoning** — and assigns 0–3 using the same rubric. Seed ~10
items whose displayed LLM grade has been deliberately corrupted, in a separate pass that *does*
show grades, and check that the dev flags them.
*Why:* trope #12, the Rubber-Stamp Effect — assessors shown LLM labels first conform to them "even
when… demonstrably incorrect," and the recommended guardrail is exactly this vigilance test
([Dietz et al. 2025](https://www.cs.unh.edu/~dietz/papers/dietz2025principles.pdf)). It also
pre-empts the obvious marker's question: *how do you know you weren't just agreeing with it?*

**12. Compute and report the full agreement table.** With stratum weights back to the Pool's grade
distribution, and 95% CIs (bootstrap over pairs) on every figure:

| Statistic | Why it's there |
| --- | --- |
| Raw exact agreement % | Transparency only — never the headline |
| Cohen's κ, 4-scale unweighted | Directly comparable to UMBRELA (0.308–0.373) and LLMJudge (~0.28) |
| **Quadratic weighted κ, 0–3** | The correct statistic for an ordinal scale (Cohen 1968) |
| Cohen's κ, binarised {0,1} vs {2,3} | Comparable to UMBRELA (0.418–0.499); the reliable part |
| Per-stratum agreement | Shows *where* the judge fails — expect grades 2 and 3 to be worst |
| 4×4 confusion matrix | Trope #5: label-level, not just system-level |
| Mean signed difference (LLM − human) | The leniency estimate; expect it positive |
| Self-consistency κ (step 8) | Drift and priming |

*Why:* chance-corrected metrics as headline, because raw agreement overstates by 10–41 percentage
points ([arXiv:2606.19544](https://arxiv.org/abs/2606.19544)); label-level analysis alongside
system-level is trope #5's guardrail; leniency is the best-evidenced IR bias (§3.1).

**13. Apply the threshold.** Pass if **binarised κ ≥ 0.40** (Landis & Koch "moderate"; the band
UMBRELA achieves against NIST). Supporting bars: 4-scale unweighted κ ≥ 0.30, quadratic weighted
κ ≥ 0.60. Below the bar, do not silently proceed — either repair (revise the query narratives,
re-judge the failing stratum by hand) or report the Arm comparison as indicative only.
In the write-up, state that McHugh (2012) would set the floor at 0.60 and that no published IR LLM
judge reaches it on a graded scale. *Why:* §4.2.

### Scoring the Arms

**14. Score every Arm on the same Qrels, with unjudged = 0.** Headline **nDCG@10** over graded
Qrels. *Why:* nDCG is defined for graded relevance
([Järvelin & Kekäläinen 2002](https://dl.acm.org/doi/10.1145/582415.582418)); unjudged-as-zero is
TREC convention.

**15. Report three robustness checks beside it.**
(a) the same metric over **binarised** Qrels — because the binarised judgments are the trustworthy
ones (§2.3);
(b) **condensed-list nDCG**, with unjudged Recipes removed, as the incompleteness check
([Sakai 2007](https://dl.acm.org/doi/10.1145/1277741.1277756));
(c) the **LOU test** — per Arm, re-score all Arms with that Arm's unique relevant Recipes removed
from the Qrels, and report the deltas
([Buckley et al. 2007](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=51236)).
If the Arm ordering is stable across all three, the conclusion is robust. If it flips anywhere,
that flip is the most interesting result in the project and leads the discussion.

**16. Do not over-claim on ~40 queries.** State the topic-set-size caveat: Voorhees and Buckley
found error rates at small topic-set sizes "larger than anticipated," and 50 topics is the TREC
working norm. At 30–50 queries, treat small nDCG differences between Arms as inconclusive and say
so rather than declaring a winner.
([SIGIR 2002](https://dl.acm.org/doi/10.1145/564376.564432))

**17. Image Query Mode runs the same pipeline with a human judge.** Same equal-depth Pool, same
0–3 rubric, Qrels hand-assigned. Report an intra-assessor consistency κ (re-judge a subset after
≥1 day, blind) instead of an LLM–human κ, and never merge the image and text agreement figures.
*Why:* premise 11; no TREC-validated multimodal prompt exists and cross-modal LLM–human agreement
is lower than text (§6).

### What the write-up must contain

**18.** The declared paradigm (step 1); the exact judge model, version and date (step 2); the full
prompt (step 3); Pool construction and depth with per-Arm contribution counts (step 5); the
agreement table with CIs (step 12); the confusion matrix; the keyword-probe result (step 9); the
LOU deltas (step 15c); and a limitations paragraph that states Soboroff's ceiling argument in his
own terms — that our Arms cannot be measured as better than the judge's notion of relevance — and
notes that circularity (Dietz tropes #1/#2) is structurally absent here because no Arm is an LLM
ranker and none shares lineage with the judge.
*Why:* Dietz et al.'s closing conditions: recently validated against human judgment, combined with
complementary metrics, circularity demonstrably mitigated, and known failure modes "acknowledged,
quantified, and addressed through appropriate guardrails."

---

## 8. Open disagreements, recorded rather than resolved

- **Can LLM Qrels replace human Qrels?** Upadhyay/Lin: yes for system-level ranking in academic
  settings. Soboroff, Clarke & Dietz: no, ever, because an answer key is a ceiling. This project
  sides with neither — it uses LLM Qrels *and* reports agreement *and* declines to call them ground
  truth. TREC's own 2026 AutoJudge track treats manual NIST assessment as ground truth and LLM
  judges as the thing being evaluated, which is the strongest available signal about where the
  community actually stands.
- **Does human-in-the-loop help?** Upadhyay et al. 2025 found hybrid regimes gave no correlation
  improvement over fully automatic — "the additional costs… do not appear to have obvious tangible
  benefits." Dietz et al. and Faggioli et al. nonetheless require human verification. The
  reconciliation: verification is not there to improve the *correlation*, it is there to make the
  claim *checkable*. That is the argument to make to a marker.
- **Pointwise vs batched judging.** Chen et al. say batching causes priming; Korikov et al. say
  batching plus self-consistency improves quality. Both are supported. At our scale the choice is
  cost-free, so take pointwise.
- **Is 0.61 the acceptable κ?** Landis & Koch say substantial agreement starts there; McHugh says
  0.60 is the minimum for any confidence; the IR literature routinely publishes 0.28–0.50 and
  defends it via Voorhees' ranking-stability result. §4.2 sets our bar at binarised κ ≥ 0.40 and
  names the stricter standard rather than hiding it.

---

## Sources

**LLM judges in IR**
[Thomas et al. 2024, SIGIR](https://arxiv.org/abs/2309.10621) ·
[Upadhyay et al. 2024, UMBRELA](https://arxiv.org/abs/2406.06519) ·
[Upadhyay et al. 2025, large-scale study](https://arxiv.org/abs/2411.08275) ·
[Soboroff 2024](https://arxiv.org/abs/2409.15133) ·
[Clarke & Dietz 2024](https://arxiv.org/abs/2412.17156) ·
[Faggioli et al. 2023, ICTIR](https://arxiv.org/abs/2304.09161) ·
[Dietz et al. 2025, ICTIR](https://doi.org/10.1145/3731120.3744588)
([PDF](https://www.cs.unh.edu/~dietz/papers/dietz2025principles.pdf)) ·
[Rahmani et al. 2024, LLMJudge](https://arxiv.org/abs/2408.08896) ·
[Rahmani et al. 2025, Judging the Judges](https://arxiv.org/abs/2502.13908) ·
[Farzi & Dietz 2025](https://arxiv.org/abs/2507.09483) ·
[TREC 2026 CFP](https://trec.nist.gov/cfp.html)

**Biases**
[Alaofi et al. 2024, SIGIR-AP](https://arxiv.org/abs/2501.17969) ·
[Chen et al. 2024, threshold priming](https://arxiv.org/abs/2409.16022) ·
[Korikov et al. 2025, batched self-consistency](https://arxiv.org/abs/2505.12570) ·
[Zheng et al. 2023, MT-Bench](https://arxiv.org/abs/2306.05685) ·
[Panickssery et al. 2024, NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html) ·
[Norman et al. 2026](https://arxiv.org/abs/2606.19544) ·
[Yu et al. 2026, overrating](https://arxiv.org/abs/2602.17170)

**Agreement statistics**
[Landis & Koch 1977](https://doi.org/10.2307/2529310) ·
[McHugh 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/) ·
[Cohen 1968, weighted κ](https://doi.org/10.1037/h0026256) ·
[Sim & Wright 2005](https://doi.org/10.1093/ptj/85.3.257) ·
[Bujang & Baharum 2017](https://riviste.unimi.it/index.php/ebph/article/view/17614) ·
[Merlo et al. 2026, ECIR](https://doi.org/10.1007/978-3-032-21289-4_27) ·
[Voorhees 2000](https://www.nist.gov/publications/variations-relevance-judgments-and-measurement-retrieval-effectiveness) ·
[Bailey et al. 2008, SIGIR](https://dl.acm.org/doi/10.1145/1390334.1390447)

**Pooling and metrics**
[Zobel 1998, SIGIR](https://dl.acm.org/doi/10.1145/290941.291014) ·
[Buckley et al. 2007](https://link.springer.com/article/10.1007/s10791-007-9032-x)
([NIST PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=51236)) ·
[Buckley & Voorhees 2004, bpref](https://dl.acm.org/doi/10.1145/1008992.1009000) ·
[Sakai 2007, Alternatives to Bpref](https://dl.acm.org/doi/10.1145/1277741.1277756) ·
[Järvelin & Kekäläinen 2002, TOIS](https://dl.acm.org/doi/10.1145/582415.582418) ·
[Voorhees & Buckley 2002, SIGIR](https://dl.acm.org/doi/10.1145/564376.564432) ·
[TREC DL 2019 guidelines](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html) ·
[TREC overview, CACM](https://cacm.acm.org/research/trec/)

**Multimodal judging**
[Hosseini et al. 2024, Zalando](https://arxiv.org/abs/2409.11860) ·
[Expanding medical case-based judgments with MLLMs, 2025](https://arxiv.org/abs/2506.17782)
