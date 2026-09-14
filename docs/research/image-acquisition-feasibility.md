# Image acquisition feasibility: 32,722 allrecipes CDN URLs

Research for [#3](https://github.com/dangth1101/IS6303/issues/3), part of the map in
[#1](https://github.com/dangth1101/IS6303/issues/1) (Milestone 1 Plan + ADRs / Milestone 2 Corpus).

Measured 2026-09-14 against the live HF datasets-server, the published Parquet export, and
`www.allrecipes.com` itself. All measurements below are first-party: HTTP responses from the
serving CDN, the dataset's own Parquet bytes, and the site's own `robots.txt` and Terms of
Service. Nothing here is taken from a secondary write-up.

## Verdict in one line

Technically cheap and fully reachable (**100/100 URLs live, ~8 GB, a few hours**), but the
rights position is **bad**: allrecipes' owner prohibits exactly this use in `robots.txt`, in
its Terms of Service, and in an `x-robots-tag: noai, noimageai` header on **every single image
response**. There is no thumbnail shortcut. Recommend the text-only fallback.

---

## 1. Permission — the blocking issue

The dataset's MIT licence covers **the CSV**, not the images. The CSV contains URL *strings*;
the JPEG bytes at the other end are owned by the site and never passed through the MIT grant.
Nobody at Hugging Face was in a position to license them. So permission has to come from
allrecipes.com, and it is refused in three independent places.

### 1a. `robots.txt` — https://www.allrecipes.com/robots.txt (fetched 2026-09-14)

The file opens with a preamble that is unusually explicit, naming ML/AI use and dataset
creation as prohibited:

```
#People Inc. content is made available for your non-commercial use subject to our
#Terms of Use here: https://www.people.inc/brands-termsofservice.
#Use of any crawler or other tool, device, or process to data mine or scrape the
#content on this website using automated means for any purpose other than
#directing traffic to this website or serving authorized advertisements on this
#website is prohibited without prior written permission from People Inc.
#Prohibited uses include but are not limited to:
#(1) text and data mining activities under Art. 4 of the EU Directive on
#Copyright in the Digital Single Market;
#(2) development or operation of any artificial intelligence, machine learning,
#or large language model (LLM) technology, including by training or fine-tuning
#such technology or using it for retrieval-augmented generation; and
#(3) creating data sets containing People Inc. content or sharing it with others.
#For the avoidance of doubt, the fact that any such tool, device, or process is
#not blocked via this robots.txt file does not constitute a waiver of any of
#People Inc.'s rights under our Terms of Use or relevant law.
```

Clause (2) covers "machine learning ... including ... using it for retrieval-augmented
generation" and clause (3) covers "creating data sets containing People Inc. content". A
multimodal retrieval index built from these images is squarely inside both. The final
paragraph pre-empts the obvious rejoinder: it says explicitly that *not* being blocked by a
`Disallow` line is **not** permission.

That rejoinder is otherwise available, and worth stating honestly. The machine-readable rules are:

```
User-agent: *
Disallow: /embed?
Disallow: /cdn-cgi/
Disallow: /cook/
```

`/thmb/` — the path all 25,407 image URLs use — is **not** disallowed for `User-agent: *`. So a
generic fetcher is not *path*-blocked. But named AI agents are blocked, and the pattern of who
is blocked is the tell:

```
User-agent: ChatGPT-User
User-agent: OAI-SearchBot
User-agent: GPTBot
Disallow: /thmb/
```

```
User-agent: Google-Extended
User-agent: anthropic-ai
User-agent: CCBot
User-agent: Claude-SearchBot
User-agent: cohere-ai
User-agent: Meta-ExternalAgent
User-agent: meta-webindexer
User-agent: PerplexityBot
Disallow: /
```

plus a further ~60-agent block (`ClaudeBot`, `Bytespider`, `ImagesiftBot`, `FirecrawlAgent`,
`archive.org_bot`, `MistralAI-*`, …) all set to `Disallow: /`. `/thmb/` is disallowed *by name*
for the AI crawlers. The only reading on which `/thmb/` is fetchable is one where we present as
something other than an AI-purpose agent — i.e. where the permission depends on
misrepresenting the purpose. That is not a defensible position.

There is **no `Crawl-delay`** directive for any agent.

### 1b. Terms of Service — https://www.people.inc/brands-termsofservice

People Inc. is the current owner of Allrecipes (formerly Dotdash Meredith;
`dotdashmeredith.com/brands-termsofservice` now redirects to the People Inc. URL, confirmed by
following the redirect). `robots.txt` names this document as the governing Terms of Use.

The live URL returns **403** to non-browser clients (Cloudflare). Text below is from the
Internet Archive snapshot `web/20260103165422` of the same URL; the document is stamped
**"Effective Date: July 31, 2025"**.

Under **"3.1 Use of the Services"**:

> Subject to this Agreement, Company grants you a limited license to use the Services solely
> for your **personal non-commercial purposes**.

Under **"3.3 Restrictions on Use of Services. You agree not to do any of the foregoing:"**

> (e) you shall not use any manual or automated software, devices, or other processes
> (including but not limited to spiders, robots, scrapers, crawlers, avatars, data mining tools
> or the like) to "scrape," harvest, or download data from the Services (except that we grant
> the operators of public search engines revocable permission to use spiders to copy materials
> from the Website for the sole purpose of and solely to the extent necessary for creating
> publicly available searchable indices of the materials, but not caches or archives of such
> materials);

> (f) you shall not use any data from the Services for the development of any software program
> (including but not limited to training a machine learning or artificial intelligence (AI)
> system);

Both clauses bite. (e) prohibits the bulk download itself. Its only carve-out is for
"operators of public search engines" building "publicly available searchable indices" — and it
excludes "caches or archives", which is precisely what a local image corpus is. A course
project's local index is not a public search engine. (f) prohibits using the data for
developing software, and calls out ML systems explicitly.

There is **no academic, educational, or research exception anywhere in the document.** I
searched for one. Clauses 3.3(a)–(g) also prohibit reproducing/distributing the Services (a),
making derivative works (d), and building a similar or competitive service (g).

### 1c. `x-robots-tag: noai, noimageai` on every image response

The strongest evidence, because it arrives with the bytes themselves. Every image response
carries a per-response, machine-readable AI opt-out:

```
x-robots-tag: noai, noimageai
cache-control: max-age=31536000,public,no-transform
server: cloudflare
```

Confirmed on **15 of 15** randomly drawn image URLs, value identical every time. `noai` and
`noimageai` are the conventional opt-out tokens for "do not use this content for AI training or
generation"; `noimageai` is the image-specific form. Note also `no-transform` in
`cache-control`.

This removes any "we didn't know" defence. The server states its position on every fetch.

### 1d. Is there a defensible academic-use position?

Honestly: **a weak one, and not one to rely on.**

- **Fair use / fair dealing** is a copyright defence, and there is a real argument that
  non-expressive analytical use of images for a retrieval experiment is transformative and
  non-substitutive. But fair use is a defence raised *after* a claim, decided case by case —
  it is not a permission you can assert in advance, and the write-up would be claiming
  compliance it doesn't have.
- Fair use does **not** answer the ToS. Downloading in breach of 3.3(e) is a contract question,
  not a copyright question, and the fair-use argument is simply unresponsive to it.
- The images are **third-party user photographs** (per ToS 2.3, users grant People Inc. a
  licence to their content). Redistributing them — e.g. committing them, or publishing them
  with the project — would compound the problem. Even under the most generous reading, images
  could never leave the local machine.
- The `noai`/`noimageai` header and the explicit anti-RAG language in `robots.txt` mean any
  breach is knowing rather than inadvertent, which is the worst posture for a graded artefact
  that will be read by an instructor.

**Position for the write-up:** we do not have permission to build an image corpus from these
URLs, and the site has refused it in three places in machine-readable form. The
`x-robots-tag: noai, noimageai` header alone should settle it.

---

## 2. Reachability — no link rot at all

The ticket assumed ~4 years of link rot since the 2022-11-15 publication. **There is none.**

| Metric | Value |
|---|---|
| Random URLs probed (HEAD) | 100 |
| HTTP 200 | **100 (100%)** |
| 4xx / 5xx / timeouts | 0 |
| `content-type` | `image/jpeg` on 100/100 |

Sampling method matters here, and my first attempt was biased. I initially sampled 100 URLs
from 8 fixed datasets-server offsets; that also returned ~100% live, but the *null-image* rate
it implied (13.25%) turned out to be badly wrong, because rows with no image cluster in blocks
(offset 11500 → 56% null, offset 14000 → 5% null — non-monotone, so the corpus is grouped by
something like category). Systematic offsets are unsafe on this dataset.

So the numbers above come from a **uniform random sample of the deduplicated URL set**, drawn
from the full Parquet export
(`https://huggingface.co/datasets/Shengtao/recipe/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet`,
31,169,113 bytes), not from paged API offsets.

The zero rot is explained by the CDN: these are Cloudflare-fronted, `max-age=31536000`
(one year) derived renditions, not links into a mutable filesystem.

### How many images actually exist

Exact counts from the Parquet export, not sampled:

| Quantity | Count | Share |
|---|---|---|
| Rows in `train` | 32,722 | 100% |
| Rows with **no** image URL (null/empty) | **7,307** | **22.33%** |
| Rows with an image URL | 25,415 | 77.67% |
| — of those, on `www.allrecipes.com/thmb/` | 25,407 | |
| — on `cf-images.us-east-1.prod.boltdns.net` (Brightcove video posters) | 6 | |
| — malformed, literal string `NaNname` (rows 11999, 23999) | 2 | |
| **Distinct URLs to fetch** | **24,759** | |

Two findings the map should absorb:

- **22.33% of recipes have no image at all.** This is much larger than the "expected link rot"
  the ticket was worried about, and it is a *dataset* property, not a network one. A multimodal
  arm would silently cover only ~77% of the corpus, which is a confound in its own right —
  independent of the licence question.
- `image` is **not unique**: the 25,413 URL-shaped values collapse to 24,759 distinct URLs (654
  duplicates), so some recipes share a photo. Worth knowing before making image identity a key.
- The two `NaNname` values are a CSV-parsing artefact and belong on the parsing-gotchas list in
  #1 alongside the `" ; "` ingredients blob and the `ast.literal_eval` instructions list.

---

## 3. Cost — cheap, and not the constraint

### Size

From `Content-Length` on the 100-URL random sample:

| Percentile | Bytes |
|---|---|
| min | 9,273 |
| p10 | 70,819 |
| p25 | 106,838 |
| **median** | **139,030** (~136 KB) |
| p75 | 397,978 |
| p90 | 918,983 |
| max | 1,277,189 |
| mean | 306,492 (~299 KB) |

The distribution is strongly right-skewed (mean is 2.2x the median), so `median x N` badly
understates the total. Because file size tracks the resolution segment in the URL, and the
corpus resolution mix is known exactly from the Parquet, the honest estimate buckets the sample
by resolution and weights by the corpus's own distribution:

| URL size segment | Sample n | Sample median bytes | Corpus URLs | Projected bytes |
|---|---|---|---|---|
| ≤300px | 7 | 13,487 | 2,229 | 0.03 GB |
| 301–800 | 2 | 109,038 | 859 | 0.09 GB |
| 801–1000 | 57 | 120,312 | 12,603 | 1.52 GB |
| 1001–1400 | 4 | 227,342 | 1,050 | 0.24 GB |
| 1401–1800 | 7 | 345,124 | 1,542 | 0.53 GB |
| 1801–2200 | 4 | 533,826 | 1,640 | 0.88 GB |
| 2201–2800 | 10 | 776,652 | 2,074 | 1.61 GB |
| 2801–3200 | 8 | 1,069,492 | 2,019 | 2.16 GB |
| 3201+ | 1 | 1,277,189 | 743 | 0.95 GB |
| **Total** | 100 | | **24,759** | **~8.0 GB** |

Cross-check: `mean x 24,759` = 7.6 GB. So **7.5–8 GB** for the full-size corpus.

Against the map's 220 GB free, storage is a non-issue — about 3.6% of the budget. It is,
however, ~125x the 64 MB CSV, so it dominates the project's disk footprint and would need to
stay out of git.

### Wall-clock time

Measured latency (HEAD, serial, from this machine): **median 1.07 s, p90 2.99 s, max 6.03 s**.
GET latency for full bytes will be somewhat higher.

| Assumed polite rate | Requests | Wall clock for 24,759 |
|---|---|---|
| 1 req/s (strictly serial, ~1 s pacing) | 24,759 | **6.9 h** |
| 2 req/s | 24,759 | 3.4 h |
| 4 req/s (e.g. 4 workers, ~1 s each) | 24,759 | **1.7 h** |

**Rate assumed for planning: 4 req/s** — a few hours, checkpointed, resumable. The map's 3-week
budget absorbs that easily. Time is not the constraint either.

### Does the CDN rate-limit or block?

Not at any rate I tested, and I deliberately did not push hard.

- Serial HEADs with 0.7 s pacing (100 requests): 100x `200`, no `429`, no throttling.
- 12-way concurrency (24 requests): **24x `200`, zero `429`**. Aggregate 1.59 req/s, but that
  figure is bounded by one 15 s outlier and by subprocess overhead in my harness, not by the
  CDN — per-request median was 1.54 s.
- No `Retry-After`, no rate-limit headers of any kind in any response.

One asymmetry worth recording: **the HTML site blocks non-browser clients, but the image CDN
does not.**

| Request | Result |
|---|---|
| `GET https://www.allrecipes.com/` with browser UA | **403** (Cloudflare) |
| `GET https://www.allrecipes.com/` with `curl/8` UA | **403** |
| `HEAD .../thmb/<...>.jpg` with `curl/8` UA | **200** |

So `/thmb/` is not UA-gated and would not need a spoofed user agent. The technical door is
open; the legal one is shut. Note that we could not even read the Terms of Service with `curl`
(403) — the site's own bot protection is a further signal about intent.

---

## 4. Shape — the size segment is **not** rewritable

This was the ticket's biggest potential cost lever, and it does not exist.

URLs have the form:

```
https://www.allrecipes.com/thmb/WLGlFdlZzoTdtUmsvHAszbh6gs0=/960x960/smart/filters:no_upscale()/3598901-quick-and-easy-brownies-...jpg
                               └── signature ──────────────┘ └─ size ┘ └crop┘ └── filters ───┘ └── source image ──┘
```

That is the signed-URL layout of **Thumbor** (`/<HMAC>=/<WxH>/smart/filters:.../<source>`) — an
HMAC over the *entire* transform path. If the size segment is part of the signed string,
changing it invalidates the signature. It is, and it does:

| Rewrite attempted | Result |
|---|---|
| `960x960` → `224x224` (and 11 other URLs) | **HTTP 400**, `text/html`, 12/12 |
| original → `320x320`, same 12 URLs | **HTTP 400**, 12/12 |
| size segment deleted entirely | **HTTP 400** |
| non-square `300x200` | **HTTP 400** |

26 rewrite probes, **zero successes**, empty response bodies. The signature is per-rendition;
each valid URL is the only URL for that rendition.

The segment does faithfully describe what is delivered — fetching two URLs and inspecting the
JPEGs confirmed `250x250` → `250x250` pixels and `1333x1333` → `1333x1333` pixels — so the
segment is a reliable *predictor* of size, just not a *dial*.

Corpus resolution mix (exact, from Parquet; 25,413 of 25,415 URLs carry a segment):

| Segment | Count |
|---|---|
| 960x960 | 12,198 |
| 250x250 | 2,228 |
| 3024x3024 | 1,527 |
| 1500x1500 | 968 |
| 2000x2000 | 897 |
| 750x750 | 672 |
| 720x720 | 646 |
| others (long tail) | 6,277 |

Median 960px, p75 1900px, max 4320px. **48% are already 960x960**, which is a convenient
consequence: for CLIP/SigLIP-family encoders wanting 224–384px input, most of the corpus is
only ~120 KB and a local downscale away.

**Implication:** there is no order-of-magnitude saving available. The ~8 GB is the price of
entry; thumbnails must be produced by downloading full size and resizing locally, which saves
disk but not bandwidth, requests, or time. It also means the `no-transform` cache directive is
the only rendition control the CDN offers.

---

## 5. Recommendation

The cost questions all came back favourable — 100% reachable, ~8 GB, ~2–7 hours, no
rate-limiting. **The licence question is the one that decides this, and it decides against.**

For the scope ticket to weigh:

1. **Text-only corpus (recommended).** Drop the image column; run the four arms on
   `title + description + ingredients` as already settled in #1. Costs nothing, since the map's
   "multimodal is for embedding" premise is an *option* on the dense arm, not load-bearing for
   the Sparse/Dense/RRF/RRF+Rerank comparison. Record the images as a documented limitation.
2. **Ask for permission.** `robots.txt` names a contact:
   `contentlicensing@people.inc`. A written academic-use grant would resolve it cleanly, but on
   a 3-week clock it is not a plan — treat any reply as a bonus, not a dependency.
3. **Do not fetch and hope.** Beyond the rights problem, the 22.33% missing-image rate means
   the multimodal arm would cover only ~77% of the corpus, introducing a confound that
   weakens the headline comparison on its own merits.

Two facts here are worth carrying into the map regardless of the image decision, because they
affect the text-only build too:

- **22.33% of rows (7,307) have no image URL** — a corpus-coverage fact, not a network one.
- **`image` is not unique** (25,415 values → 24,759 distinct) and two values are the literal
  string `NaNname` (rows 11999, 23999) — a new parsing gotcha.

---

## Method and caveats

- **Read-only sample. No bulk download was started**, per the ticket. Total traffic: ~265 HEAD
  requests plus 2 small GETs (~25 KB) to verify pixel dimensions, paced with 0.6–0.7 s delays.
- **Counts** (32,722 / 7,307 / 25,415 / 24,759 / resolution mix) are **exact**, computed over
  the full Parquet export — not extrapolated.
- **Reachability and size** rest on a 100-URL uniform random sample of the 24,759 distinct
  URLs. At 100/100 live, the 95% one-sided binomial lower bound on the live rate is 97.1%; the true rate
  could plausibly be a few percent below 100%, but wholesale link rot is ruled out. Size
  percentiles from n=100 are indicative; the bucket-weighted total is the more reliable figure
  because it leans on the exact corpus resolution mix and only samples within buckets — though
  the sparse buckets (n=1–4 for 300–800px, 1001–1400px, 1801–2200px, 3201+px) are the weakest
  links in that estimate.
- **Latency** was measured from one machine on one network at one time of day; treat wall-clock
  projections as order-of-magnitude.
- **The ToS text is from an Internet Archive snapshot** (`web/20260103165422`) because the live
  page returns 403 to non-browser clients. The snapshot is self-stamped "Effective Date: July
  31, 2025". If the licence position ever becomes load-bearing, re-read the live page in a real
  browser and confirm the clause numbering hasn't moved.
- **Not a lawyer, and this is not legal advice.** What is reported is what the sources say
  verbatim, with the clause references so the judgement can be re-made by someone else.
