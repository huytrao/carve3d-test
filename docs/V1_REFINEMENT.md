# V1 quality refinement from the completed T4 x2 run

## Supplied baseline result

Lower MRC is better. The completed v8 run was numerically stable, but its
generalization gain was very small:

| Metric | Before RL | Best / after RL | Improvement |
| --- | ---: | ---: | ---: |
| Held-out validation MRC | 0.214209 | 0.212425 at epoch 5 | 0.001784 (0.833%) |
| Held-out validation MRC at epoch 55 | 0.214209 | 0.213377 | 0.000832 (0.388%) |
| Final same-seed mean MRC | 0.213262 | 0.213185 | 0.000077 (0.036%) |

The four post-RL final MRC values were `0.183030`, `0.235778`, `0.159612`, and
`0.274321`. The last update remained stable (`grad_norm=0.000851`,
`replay_error=3.55e-4`, KL approximately `1e-5`), so this was not a crash or
gradient explosion. The limiting issue was weak transfer to the final target.

The old runner restored `paper_last_safe` at epoch 55 even though epoch 5 had
the lowest validation MRC. That setting follows the paper's KL-stop convention,
but it is not the best choice for this small, noisy T4 experiment.

## V1 changes

V1 is a target-specific continuation, not a claim of a new paper reproduction:

- Load v8 `checkpoints/best_lora.pt` as the starting policy. In the supplied
  run, this is epoch 5 rather than epoch 55.
- Treat that loaded LoRA as the no-regression baseline. If no continuation
  checkpoint lowers held-out validation MRC, V1 returns the initial epoch-5
  LoRA.
- Replace unrelated machine/crane prompts with ten disjoint descriptions of
  stairs, stepped platforms, and open braced frames. Four other staircase
  descriptions select the checkpoint; the exact final prompt remains held out.
- Retain eight same-prompt trajectories. Reducing to four would be faster but
  would make the small-batch advantage estimate noisier.
- Use `3e-5` instead of `7.5e-5` for continuation, validate every three epochs
  with four fixed seeds per validation prompt, select `best_validation`, and
  stop after five validation rounds without improvement.
- Skip the 100-prompt curation pass because the V1 prompt family is fixed. This
  removes roughly 32 minutes from the supplied timing without weakening each RL
  update.
- Write `training_summary.md` automatically alongside `rlft_metrics.json`.

No method can guarantee a lower unseen-test MRC from a single stochastic run.
V1's guarantee is narrower and testable: it will not replace the supplied best
LoRA unless the fixed, held-out validation suite improves by at least `1e-4`.

## Starting V1 on Kaggle

Keep the previous checkpoint at:

```text
/kaggle/working/carve3d-paper-style-t4x2-output-v8/checkpoints/best_lora.pt
```

If the old Kaggle session has ended, save/attach that output as a Kaggle Dataset
and set the path before running `code.txt`:

```python
import os
os.environ["CARVE3D_V1_INITIAL_LORA"] = "/kaggle/input/<dataset>/checkpoints/best_lora.pt"
```

The V1 output is written to
`/kaggle/working/carve3d-staircase-refinement-t4x2-v1`.
