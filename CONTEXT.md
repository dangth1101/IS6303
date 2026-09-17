# IS6303

Hybrid retrieval over a recipe corpus, answerable by text or by image, with a
thin generation layer over the results. The project's output is the justified
choice of retrieval design, so the language below is what the comparison is
argued in.

## Language

### The corpus

**Recipe**:
One dish with its ingredients, directions and metadata. The thing a user is
looking for, and the unit that gets ranked and judged.
_Avoid_: Document, record, item

**Corpus**:
The full set of Recipes the system can retrieve.
_Avoid_: Dataset (that is the file on disk, before it becomes a Corpus)

**Chunk**:
A contiguous piece of one Recipe's text, indexed and retrieved on its own. A
Recipe has many Chunks; a Chunk belongs to exactly one Recipe.
_Avoid_: Passage, segment, document

**Chunking Strategy**:
A rule for dividing a Recipe into Chunks. Interchangeable within one pipeline,
so that two indexes differ only by it.
_Avoid_: Splitter, chunker

**Retrieval Text**:
The text of a Chunk that is embedded or indexed, as distinct from the text shown
to a user. The two need not be the same.
_Avoid_: Content, body

### Retrieving

**Arm**:
One complete retrieval configuration, run end to end so its ranking can be
compared against another's. Keyword, vector and their Fusion are each an Arm.
_Avoid_: Mode, strategy, variant, pipeline

**Fusion**:
Combining the rankings of two or more Arms into one ranking.
_Avoid_: Blending, merging, hybrid (hybrid describes the system, not the step)

**Roll-up**:
Collapsing the Chunk hits of a query into a ranking of Recipes, since Chunks are
retrieved but Recipes are returned.
_Avoid_: Aggregation, grouping, dedup

**Query Mode**:
Whether a query arrives as text or as an image. Both return ranked Recipes.
_Avoid_: Input type, modality

**Cross-modal Retrieval**:
Matching an image query against Recipe text held in a shared embedding space,
without embedding Recipe images.
_Avoid_: Image search, visual search (both imply matching image to image)

### Judging

**Qrel**:
A graded judgment that a given Recipe is or is not relevant to a given query.
The ground truth every Arm is scored against.
_Avoid_: Label, annotation, rating (a Recipe's `rating` is its user score, an
unrelated thing)

**Pool**:
The union of the top results from every Arm for one query — the set that gets
judged, so that no Arm is scored on Recipes the others never surfaced.
_Avoid_: Candidate set, result set

**Generation Layer**:
The pass that turns retrieved Recipes into prose for the user. It reads the
ranking; it never changes it.
_Avoid_: RAG (that names the whole system), synthesis, answering
