---
name: speciesnet-taxonomy-mapping
description: Build a SpeciesNet taxonomy mapping CSV from a user's custom camera-trap category labels, for use with MegaDetector's restrict_to_taxa_list(). Use when asked to prepare/create/build a SpeciesNet mapping file, map a study area's species or label list to the SpeciesNet taxonomy, or restrict SpeciesNet model outputs to a custom set of categories.
---

# SpeciesNet taxonomy mapping file

## What this skill does

This skill turns a user's list of camera-trap category labels (the tags they actually use for one study area) into a SpeciesNet **mapping CSV** that can be consumed by MegaDetector's `restrict_to_taxa_list()` (https://megadetector.readthedocs.io/en/latest/postprocessing.html#megadetector.postprocessing.classification_postprocessing.restrict_to_taxa_list). SpeciesNet (https://github.com/google/cameratrapai) predicts across ~3000 categories (mostly species, some higher taxa). `restrict_to_taxa_list()` collapses those predictions down to the user's much smaller label set by walking up the taxonomic tree, matching the most specific applicable row in the mapping file. Our job is to author that mapping file: one row per user label, telling the function which SpeciesNet taxon each label corresponds to.

## Inputs you need (ask if missing)

- **Input file**: a path to the user's label list. Usually a one-label-per-line `.txt`/`.csv` with no header (e.g. `Baboon_chacma`, `Buffalo_cape`), sometimes a CSV with a header. Read it as-is; the user's tokens become the `original_common` column verbatim.
- **Output file**: the path to write the mapping CSV. If the user has a `taxonomy-lists/` folder, that is the usual home; mirror existing filenames (e.g. `<project>_speciesnet.csv`).
- **Country / state / region**: ask for this if the user doesn't provide it — it resolves many ambiguities (a "lion" tag means *mountain lion* in Idaho but *Panthera leo* in Tanzania; a "robin" means different birds in North America vs Europe). If the user truly has no region, ask whether the mapping is meant to be global or region-specific.
- **Scope** (ask if unclear): by default, assuming this mapping is for a **specific study area / reserve** (treat the list as the full set of taxa present — fold predictions of unlisted species into the nearest listed taxon); only if the user expressly indicates that the list is non-exhaustive should you treat this mapping as applying to **a whole country/region** (unlisted species may genuinely occur; be more conservative about absorbing them)? Specific-study-area is the common case and licenses more aggressive mapping.

## Locating the SpeciesNet taxonomy

You need `taxonomy_release.txt`. Check these paths in order, and only download if neither exists:

1. `c:/git/cameratrapai/data/model_package/taxonomy_release.txt`
2. `~/git/cameratrapai/data/model_package/taxonomy_release.txt`
3. Download from `https://raw.githubusercontent.com/google/cameratrapai/refs/heads/main/data/model_package/taxonomy_release.txt`

It is ~350KB (~3000 lines) and fits in memory. Each line is `guid;class;order;family;genus;species;common`, e.g. `aa73e0ac-...;mammalia;carnivora;felidae;panthera;pardus;leopard`. The GUID is irrelevant. Empty trailing fields denote higher taxa: a genus row is `...;felidae;panthera;;panthera species`, a family row is `...;felidae;;;felidae family`, an order row is `...;carnivora;;;;carnivora order`, a class row is `mammalia;;;;;mammal`. For almost every taxon, its parent taxa also appear as their own rows, so you can rely on family/order/class rows existing.

## Output format

Write a CSV with this header: `latin,common,original_latin,original_common,notes`.

- **latin** (required, but may be blank): the SpeciesNet taxon this label maps to, written as the *most-specific token* of the matched row — a species is `genus species` (e.g. `panthera pardus`), a genus is just `genus` (e.g. `panthera`), a family is `family` (e.g. `felidae`), an order is `order` (e.g. `carnivora`), a class is `class` (e.g. `mammalia`). Every non-empty value MUST be a real SpeciesNet category (validate at the end — see below). Leave blank when there is no good mapping.
- **common**: the SpeciesNet common name corresponding to the represented taxon — use the *species'* common name when a higher-rank `latin` stands in for a single species (e.g. `latin=aepyceros, common=impala`), and the group/common name when the mapping is a genuine group or catch-all (e.g. `latin=chiroptera, common=bat`). This column is largely cosmetic: `restrict_to_taxa_list()` is essentially always called with `use_original_common_names_if_available=True`, so `original_common` overrides it for display.
- **original_latin**: optional bookkeeping (the Latin name the *user* uses). Usually left blank.
- **original_common**: the user's own token, verbatim. Populate this for every row that comes from the input list. This is the label the user will actually see in output.
- **notes**: free-text for the human reviewer. `restrict_to_taxa_list()` ignores it. Use it to flag surprises, judgment calls, and taxa you couldn't map.

CSV quoting: wrap any field that contains a comma (most `notes` do) in double quotes, doubling internal quotes. Apostrophes don't need quoting. Latin tokens contain spaces but never commas. Preserve the input order of labels so the user can cross-check against their list. A bundled real example for South Africa lives next to this skill at `ppp-botha_speciesnet.csv` — consult it for the exact format and for how each tricky case below was handled.

## The core principle: map at the highest level that creates no ambiguity

You cannot decide rows one at a time. The level you map each label to depends on what *else* is in the list. The rule: for each label, climb the taxonomic tree to the **highest ancestor whose subtree contains no other label's target taxon**, then use that ancestor as `latin`. Concretely:

1. First, determine the true biological identity and the corresponding SpeciesNet taxon for *every* label (the "target taxon"). Use domain knowledge, grep the taxonomy for common/Latin names, and web-search to reconcile nomenclature when the user's name doesn't match SpeciesNet's (this is common; it's easier when common names are available).
2. Then, for each label, walk up from its target taxon and stop just before reaching a taxon that is also an ancestor of a *different* label's target. That stopping point is the `latin` value.

Worked consequences (all from the taxonomy, not memory):

- If the list has `red_fox` and `bobcat` and no other canids/felids, map `canidae` -> red fox and `felidae` -> bobcat (each family has only one representative in the list), NOT the species. If it also had `vulpes` and no other carnivores, map the whole order `carnivora` -> vulpes.
- If the list has three species in one genus (e.g. bushbuck, nyala, greater kudu — all `tragelaphus`), you cannot use the genus; map each at the species level (`tragelaphus scriptus`, `tragelaphus angasii`, `tragelaphus strepsiceros`).
- If a genus has exactly one representative in the list, map at the genus (e.g. only `aepyceros melampus` -> map `aepyceros` -> impala), which also folds any other species in that genus into the label.  Ditto for family and order level.  **It will be very common to map labels to the genus level**.  Family and even order-level mappings will also be quite common.
- This is independent of whether common or Latin names were provided; apply the same "highest unambiguous level" logic either way.
- You can also create deliberately nested rows when the user has an "other" bucket, because the most specific matching row wins: `odocoileus,white-tailed deer` then `cervidae,other deer` sends the genus Odocoileus to "white-tailed deer" and all remaining deer to "other deer".

### Catch-all labels

Labels like `Mammal_unspecified`, `Bird_unspecified`, `Rodent_unspecified`, or plain `bird` / `mammal` / `reptile` are intentional high-level buckets. Map them straight to the high taxon (`mammalia`, `aves`, `rodentia`, `reptilia`, `chiroptera`, `insecta`, etc.). Do **not** let them block the specific labels: because the most-specific match wins, specific birds still resolve at their own level and only the leftovers fall through to the catch-all. When computing collisions for the specific labels, ignore the catch-alls (a specific taxon may correctly sit *inside* a catch-all's subtree).

### Aggressiveness and the notes column

Default to **aggressive** mapping for a specific study area: climb to the highest unambiguous level even when the broader taxon contains other species that occur in the region but aren't in the list — the list defines the study area, so off-list predictions are best folded into the nearest listed taxon. BUT add a `notes` entry whenever the fold is **surprising** — i.e. it absorbs a species that is common/iconic in the region and whose absence from the list is genuinely unexpected. Example: in South Africa, mapping `panthera` -> leopard folds lions into "leopard"; lions are common there, so note it and ask the user to confirm lions are truly absent. Do not over-note unremarkable folds (e.g. folding korhaans into "kori bustard" needs no note).

## Special cases

- **Non-taxonomic entities** — `Human`, `Vehicle`, `Blank` (also `empty`, `unknown`, `fire`, `car or truck`, etc.): include the row but leave `latin` (and `common`) **blank**. Do not map Human to `homo sapiens`; treat it as a non-taxonomic category.
- **Sex/age variants** — `lion`, `lion_male`, `lion_female`, `lion_cub` all map identically, as if the user had just written the species. Same for tracks/scat-style qualifiers — map to the species.
- **Species absent from SpeciesNet but with a parent that exists** — map to the nearest available parent and note it. E.g. a Mexican long-nosed armadillo (not in the taxonomy) with no other armadillos in the list -> `latin=dasypodidae`, `original_common=Mexican long-nosed armadillo`. Or Kirtland's warbler (absent) under "unknown perching bird" -> `passeriformes`. Put the user's name in `original_common`.
- **Species absent from SpeciesNet with no usable parent** — if the genus is absent *and* the family already contains other listed labels (so you can't climb without ambiguity), leave `latin` blank and explain in `notes` (e.g. Selous' mongoose, *Paracynictis selousi*: genus not in the taxonomy and Herpestidae has other listed mongooses).
- **Tokens with no mapping at all** — still emit the row with a blank `latin`.
- **Multiple rows mapping to the same output name** — this is allowed and sometimes the right move, for two reasons: (a) to **exclude** part of a parent genus. Example: a list with `Zebra` and `Domestic_donkey` but no horse — rather than mapping genus `equus` -> Zebra (which would fold in *domestic horse*, which looks nothing like a zebra), list the zebra species explicitly: `equus quagga` -> Zebra and `equus zebra` -> Zebra, leaving horse to fall through. (b) to capture a **non-taxonomic but visually coherent concept** that spans several taxa. Example: "bird of prey" is a clear visual class but not one taxon, so emit several rows mapping to the same output name (e.g. `accipitridae`, `falconidae`, `strigiformes`, `sagittariidae`, ... -> "bird of prey").
- **Nomenclature mismatches** — users often paste Latin names that don't match SpeciesNet's taxonomy (synonyms, recent splits/lumps, genus reassignments like *Galerella* vs *Herpestes*, *Lupulella* vs *Canis*). Web-search to reconcile, and prefer the name that exists as a SpeciesNet category.

## When unsure

The user always reviews the output and can fill in a few cases by hand, so **bias toward caution**: when a mapping is genuinely ambiguous, leave `latin` blank and add a `notes` line rather than guessing. If you are ~98% confident, populate `latin` and leave a `notes` line explaining the call. It is much better to under-commit with a note than to silently guess wrong.

## Validate before finishing

After writing the file, programmatically confirm every non-empty `latin` is a real SpeciesNet category — this is exactly the check `restrict_to_taxa_list()` performs, and it catches typos and stale names. Run:

```python
import csv
tax_path = "C:/git/cameratrapai/data/model_package/taxonomy_release.txt"  # or wherever it resolved
valid = set()
with open(tax_path, encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\n").split(";")
        if len(p) < 7:
            continue
        _, cls, order, family, genus, species, _ = p[:7]
        if species:   valid.add(f"{genus} {species}".strip())
        elif genus:   valid.add(genus)
        elif family:  valid.add(family)
        elif order:   valid.add(order)
        elif cls:     valid.add(cls)

with open("OUTPUT.csv", encoding="utf-8") as f:
    bad = [(r["original_common"], r["latin"]) for r in csv.DictReader(f)
           if r["latin"].strip() and r["latin"].strip() not in valid]
print("invalid latin values:", len(bad))
for oc, lat in bad:
    print("  BAD:", repr(lat), "<-", oc)
```

Fix any reported rows (usually a nomenclature mismatch — search the taxonomy/web for the correct token) until the count is zero.

## End-to-end procedure

1. Get input path, output path, and region (ask if missing); confirm scope (specific study area vs whole region) if it matters.
2. Locate / download `taxonomy_release.txt`.
3. Read the user's label list verbatim.
4. For each label, identify the species/concept and its SpeciesNet target taxon (grep the taxonomy; web-search to reconcile names; apply the region to disambiguate).
5. Across the whole list, compute the highest-unambiguous-level `latin` for each label; handle catch-alls, non-taxonomic entities, sex/age variants, absent-species, and multi-row cases per the rules above.
6. Double-check each mapping to make sure there isn't an opportunity for a higher-level mapping, in particular check that you didn't map species when you could have unambiguously mapped a genus.
7. Write the CSV (`latin,common,original_latin,original_common,notes`), preserving input order, quoting fields with commas, adding `notes` for surprises and judgment calls.
8. Run the validation script; fix until zero invalid `latin` values.
9. Tell the user what you wrote, and call out the rows you flagged in `notes` (surprising folds, unmapped taxa) so they know what to review.
