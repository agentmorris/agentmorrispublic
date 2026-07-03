---
name: animal-category-mapping
description: Build a best-effort CSV that maps one wildlife/camera-trap dataset's category names (source) onto another's (target), where both category lists are given as COCO-format .json files, so the source dataset can be used for quantitative evaluation of a model trained on the target categories. Use when asked to map, align, or crosswalk animal/species categories between two labeled datasets, or to produce a source-to-target category mapping CSV.
---

# Animal category mapping

Produce a "best effort" mapping from the categories used in a *source* dataset to the categories used by a *target* model, so the source (which was labeled with its own category scheme) can be scored against the target model's predictions. A perfect mapping usually does not exist: some source categories map cleanly, some collapse many-to-one, some map only to a coarser target class, and some have no reasonable target at all.

## Inputs and output

The user provides three things:

1. A source category file in COCO `.json` format (you only need the category names).
2. A target category file in COCO `.json` format (you only need the category names).
3. A path for the `.csv` file you will create, with columns `source_category`, `target_category`, and (optionally) `notes`.

Write one row per source category, in the source file's original category order. Columns:

- `source_category`: the source category name, verbatim.
- `target_category`: the chosen target category name, verbatim, or left empty if there is no reasonable mapping.
- `notes`: fill this in for anything that was a judgment call, so the user can find and override it quickly. Leave empty for obvious 1:1 mappings.

## Do not read the .json files into context

These COCO files are large (often hundreds of KB to many MB of annotations) and reading the whole file wastes context and can exceed tool limits. You only need the `categories` array. Extract just the category names with a tool instead of reading the file. For example:

```python
import json
for path in [SOURCE_JSON, TARGET_JSON]:
    with open(path) as f:
        d = json.load(f)
    names = sorted(c['name'] for c in d['categories'])
    print('===', path, '(%d categories)' % len(names))
    for n in names:
        print(repr(n))
```

## Mapping principles

- Best effort, not exhaustive. It is expected that many target categories go unused (e.g. mapping an eastern-US species list onto a California model). That is fine and worth a one-line note to the user, not a problem to solve.
- Naming differences are the easy case: `mouse` vs `mouse species`, `other_rodent` vs `rodent_other`, `false_trigger` vs `blank`. Map these directly.
- Many-to-one is common: e.g. `brown_rat` and `black_rat` both map to a single `rat` target.
- Coarser target classes: if the source has `turtle` and the target has only `other_reptile`, map to `other_reptile`. If the target has no turtle and no generic reptile class, leave it blank.
- Empty/non-animal images: source datasets usually have one of `blank`, `empty`, `misfire`, `false trigger`, `false_trigger`, `no_animal`; these all map to the target's equivalent (`blank`, `empty`, etc.). If the source distinguishes human/camera-servicing images (`setup`, `pickup`, `person`) and the target has a matching class (e.g. `setup_pickup`), map accordingly; otherwise treat by best fit or leave blank.
- Use real taxonomy knowledge to make the closest honest match, and note it. Examples encountered in practice: a massasauga is a rattlesnake (map to `rattlesnake`, not generic `snake`, if both exist); a mink is a mustelid (closest to `weasel`); a woodchuck/groundhog is in the squirrel family Sciuridae but looks nothing like a tree squirrel; jumping mice (Zapodidae) are not true mice; a sora is a bird (a rail). Shrews and moles are frequently combined into one target class (`shrew_mole`).
- When in doubt, leave `target_category` empty rather than guess. A blank is more useful for evaluation than a wrong mapping.

## Ask the user about genuine judgment calls

The user has said they are never in a hurry and prefer to be asked rather than have you guess. Do the confident mappings yourself, then collect the genuine judgment calls and ask about them (a batched multiple-choice question works well). Wait for the actual answer; do not treat elapsed time as permission to proceed. Typical judgment calls worth asking about:

- A source animal in a family that the target splits or lumps differently (e.g. woodchuck to `squirrel` vs `mammal_other`).
- A source animal with no exact target class where the choice is "generic catch-all class" vs "leave blank" (e.g. `raccoon` to `mammal_other` vs blank).
- A source class that is more specific than any single target class, where more than one coarse target is defensible.

Put the resulting decision in the `notes` column so it is easy to revisit.

## Writing and validating the CSV

Use a CSV writer with proper quoting: category names can contain commas and apostrophes (e.g. `kirtland's_snake`). Before finishing, validate:

- Every source category appears exactly once.
- Every non-empty `target_category` value is actually present in the target category set (guards against typos and hallucinated target names).
- Report which source categories were left blank and which target categories went unused, so the user can sanity-check.

```python
import json, csv

with open(TARGET_JSON) as f:
    target_names = {c['name'] for c in json.load(f)['categories']}
with open(SOURCE_JSON) as f:
    source_cats = sorted(json.load(f)['categories'], key=lambda c: c['id'])
source_order = [c['name'] for c in source_cats]

# mapping: source_name -> (target_name_or_empty, note)
mapping = { ... }

assert set(mapping) == set(source_order), 'source coverage mismatch'
bad = [(s, t) for s, (t, _) in mapping.items() if t and t not in target_names]
assert not bad, 'target names not in target set: %s' % bad

with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['source_category', 'target_category', 'notes'])
    for name in source_order:
        t, note = mapping[name]
        w.writerow([name, t, note])
```

## Example

`example-osu-to-california-mapping.csv` in this skill folder is a completed mapping from an Ohio (eastern-US) small-animals dataset (46 categories) onto a California small-animals model (29 categories). It illustrates the conventions above: many-to-one snake and bird collapses, `empty` to `blank`, taxonomy-based calls (massasauga to `rattlesnake`, mink to `weasel`), two turtles left blank because the target has no turtle or generic-reptile class, and `notes` used on every judgment call.
