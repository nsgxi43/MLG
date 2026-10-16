# MLG walkthrough — 4 minutes 30 seconds

Use the 17-slide outline in SLIDES.md as supporting material; the short walkthrough emphasizes slides 1, 5–9, 11–16. Open existing files before presenting; no experiment needs to run live.

**0:00–0:30 — Problem and research.** “We classify molecular graphs from atom labels and connectivity. We studied Shervashidze et al.’s WL kernels and Rieck et al.’s persistent WL procedure. Our question is whether component information, cycle magnitudes, or cycle presence changes predictive performance.” MUTAG is primary; completed PTC-MR results are supplementary.

**0:30–1:05 — WL and P-WL.** “WL refines each node label with the multiset of neighbor labels and concatenates counts across depths. Their dot product defines a kernel; our classifier instead receives explicit vectors in a Random Forest. P-WL uses label distances to order existing edges. Merges produce component persistence features.” Show `wl.py` and the pipeline on slide 10.

**1:05–1:40 — Cycles.** “Edges whose endpoints are already connected close cycles. P-WL-C sums their birth-weight powers by both endpoint labels in a separate block. Triangles remain cycles: we do not fill them. This is a paper-derived enhancement, and it improved MUTAG accuracy by 1.61 points over P-WL, even though WL remained higher. On PTC-MR, cycles reduced accuracy by 0.55 points.” Show the existing ablation evidence.

**1:40–2:10 — λ attempt and binary change.** “Our first adaptation scaled the cycle block. Zero exactly reproduces P-WL. All tested positive weights gave identical predictions and selected parameters to P-WL-C. Trees can be insensitive to positive rescaling because ordering is unchanged. Our primary new modification instead maps each completed cycle coordinate to presence: nonzero becomes one, zero stays zero. We leave every vertex value, key, dimension, and ordering unchanged.” Show `pwl.combine_binary` and slide 9.

**2:10–2:40 — Fair evaluation.** “We reused the original 100 MUTAG outer splits and their five inner folds. Inner accuracy selects depth 0–6, p=1 or 2, and 25/50/100 trees. Vocabularies fit on training data only. Accuracy and macro-F1 are averaged within each repeat, then summarized across ten repeats. Baselines and persistence caches were preserved.” Show verification and slide 11.

**2:40–3:35 — Actual results.** P-WL-C-Binary: accuracy 86.06 ± 1.16%; macro-F1 84.30 ± 1.60%. Versus P-WL-C: accuracy -1.71637 pp; macro-F1 -1.98133 pp. “The highest mean accuracy is WL; the highest mean macro-F1 is WL. Cycles are useful relative to P-WL on MUTAG, while method performance differs on PTC-MR.” Show `results/binary-cycle/figures/MUTAG_accuracy_comparison.png` and `MUTAG_macro_f1_comparison.png`.

**3:35–4:05 — Paired changes.** “Accuracy: 17 improved folds, 41 worsened, 42 tied. Predictions changed on 67 folds, and selected parameter triples changed on 96. These are descriptive paired results, not significance evidence.” Show `MUTAG_paired_fold_differences.png`.

**4:05–4:30 — Lessons and limits.** “Binary encoding changed predictions and reduced mean accuracy. Cycle magnitudes may carry useful information; this is an interpretation, not a causal result. Joint retuning also changed some selected parameters, so the measured difference is between nested method pipelines, not a fixed-model intervention. Small datasets and overlapping repeats limit inference. Our project implements, modifies, and measures representations; it does not promise every change improves performance. PTC-MR-Binary was not evaluated: out of scope.”

For validation before the presentation: `.venv/bin/python -B -m unittest -v test_binary`. To regenerate verified case-study reports and new figures without training: `.venv/bin/python -B finalize_case_study.py`. These write only the case-study documents/new binary analysis outputs; they preserve `results/full/`. Keep the baseline `run.py --report-only` command out of this workflow because it regenerates the older four-method documents.
