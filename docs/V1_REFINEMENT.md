# V1 from-scratch convergence run on Kaggle T4 x2

## Why the previous run did not look converged

Lower MRC is better. The completed run was stable but changed the policy only
slightly:

| Metric | Before RL | Best / after RL | Improvement |
| --- | ---: | ---: | ---: |
| Held-out validation MRC | 0.214209 | 0.212425 at epoch 5 | 0.001784 (0.833%) |
| Held-out validation MRC at epoch 55 | 0.214209 | 0.213377 | 0.000832 (0.388%) |
| Final same-seed mean MRC | 0.213262 | 0.213185 | 0.000077 (0.036%) |

The four post-RL final MRC values were `0.183030`, `0.235778`, `0.159612`, and
`0.274321`. The last update remained numerically stable
(`grad_norm=0.000851`, `replay_error=3.55e-4`, KL approximately `1e-5`). This
was not a crash or exploding-gradient failure. However, KL remaining close to
zero and only `0.036%` final transfer show that the policy barely moved in a
useful direction.

The old output also selected `paper_last_safe` at epoch 55 although epoch 5 had
the best held-out MRC. For a small T4 run, that makes the final model harder to
interpret.

## V1 experiment

V1 deliberately starts from base MVDream with zero LoRA. It does not load any
checkpoint from the previous run.

| Setting | V1 value | Reason |
| --- | ---: | --- |
| Paper epochs | 61 | One statistics warmup plus at most 60 optimizer updates |
| Batch | 8 trajectories, one prompt | Preserve a usable per-prompt advantage estimate |
| Training prompts | 10 staircase/stepped-frame variants | Optimize geometry relevant to the requested result |
| Prompt visits | 6 per prompt | Twice the target-aware exposure of the earlier 30-update proposal |
| Initial LR | `1.5e-4` | Move farther than the previous conservative `7.5e-5` run |
| LR plateau rule | halve after 2 non-improving validations | Anneal instead of continuing forever at one LR |
| Minimum LR | `1e-5` | Allow fine convergence after the larger initial updates |
| Validation | every 5 epochs, 4 prompts x 4 fixed seeds | Reduce checkpoint-selection noise |
| Plateau stop | 10 non-improving validations | Stop after LR annealing has had time to work |
| Checkpoint | `best_validation` | Never publish a later checkpoint merely because its KL is safe |
| KL guard | `3.2e-4` | Discard/stop before excessive policy drift |

The ten training prompts, four validation prompts, and exact final prompt are
disjoint. The final prompt never selects a checkpoint. Curation is skipped
because the prompt family is already target-specific; the saved time is spent
on twice as many useful optimizer updates.

The timestep loss remains a mean. The released trainer accumulates gradients
across diffusion timesteps and effectively averages them; changing V1 to a sum
would only multiply gradient magnitude and could create unstable, misleading
movement rather than better convergence.

## Reading convergence

`training_summary.md`, `rlft_metrics.json`, and `training_progress.json` are
written under:

```text
/kaggle/working/carve3d-staircase-from-scratch-t4x2-v1
```

Use all of these conditions rather than training MRC alone:

1. Best held-out validation MRC must be below the base-policy validation MRC.
2. The best should persist across fixed seeds instead of appearing in one
   training batch.
3. Learning rate should decay when validation plateaus; a later lower-LR
   checkpoint should either improve or trigger early stopping.
4. KL must remain below the guard, gradients must stay finite, and replay error
   must remain small.
5. Same-seed final mean MRC should also decrease. This last value is a test, not
   a checkpoint-selection signal.

A single stochastic T4 run cannot guarantee a lower unseen-test MRC. The V1
control flow guarantees that the published checkpoint is the lowest fixed-seed
held-out validation checkpoint seen during the from-scratch run, with base
MVDream retained if no update improves it.

`code.txt` removes only the previous V1 output directory before launching so a
second execution is also a genuine zero-LoRA run. It does not delete the old v8
result.
