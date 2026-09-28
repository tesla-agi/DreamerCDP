# dreamer-cdp

A from-scratch PyTorch reimplementation of **Dreamer-CDP** (Hauri & Zenke, 2026): DreamerV3 with the pixel decoder removed, trained instead to predict the encoder's own embedding of each frame from the recurrent state. It runs on Crafter on a single Apple-silicon Mac.

![dreamer-cdp architecture](docs/architecture.svg)

## Why

DreamerV3 already imagines entirely in latent space: its rollouts never produce a pixel. But it learns that latent space by reconstructing pixels, so the representation is shaped by what can be redrawn rather than by what can be predicted, and in Crafter most of what can be redrawn is grass. Dreamer-CDP keeps Dreamer's architecture (the RSSM, imagination, and the actor-critic trained inside it) and changes only the training signal. Dreamer supplies the architecture; a JEPA-style objective supplies the learning signal.

## What changes relative to DreamerV3

One term of the world-model loss is deleted and one is added. Everything else is unchanged.

```text
DreamerV3     L = −log p(o_t | h_t, s_t)            + L_reward + L_continue + β_dyn·KL_dyn + β_rep·KL_rep
dreamer-cdp   L = −β_cdp · cos( g(h_t), sg(e_t) )   + L_reward + L_continue + β_dyn·KL_dyn + β_rep·KL_rep
```

Here `e_t` is the encoder's output for frame `t`, `sg` is stop-gradient, and `g` is a small MLP (`600 → 400 → 400 → 400 → 6144`). In code the target is simply `embed.detach()`: one encoder, one forward pass, used twice.

## Design decisions

**The predictor reads `h_t` and nothing else.** `h_t = GRU(h_{t−1}, s_{t−1}, a_{t−1})` is computed before frame `t` is observed. The stochastic state `s_t` is sampled from the posterior `q(s_t | h_t, e_t)`, so it has already seen the frame, and a predictor given `s_t` could read the answer out of its own input. The old decoder was allowed `(h_t, s_t)` because its target *was* the frame; the predictor's target is derived *from* the frame, so any post-observation input is a shortcut. The smoke test in `world_model.py` checks this directly: it perturbs frame 5 and asserts that `h_5` does not change.

**Cosine rather than L1 or L2.** The target comes from the same encoder that is being trained. Under L1 or L2 the model can lower the loss by shrinking every embedding without improving any prediction: scale everything by 0.1 and L2 falls 100×, while the prediction points exactly as wrong as before. Cosine is scale-invariant, which closes that path. I-JEPA and V-JEPA close the same loophole from the other side, by normalizing the target and then using L2 or L1.

**No EMA target network.** Following the paper and Tang et al. (2023), collapse is prevented with two timescales instead: the RSSM and predictor learn at 4·10⁻⁴ and the encoder at 6·10⁻⁶, so the predictor stays close to its optimum while the representation moves slowly. An EMA slows the target down; a fast predictor produces the same separation from the other end.

**The first step of each sequence is excluded** from the cdp loss, because `h_0` comes from the initial state and has seen nothing yet.

## Diagnostics

A rising cosine on its own is misleading. Crafter frames share a large common component, mostly terrain, so early in training the cosine climbs to around 0.98 while the model has learned nothing beyond "a typical screen". Every log line therefore reports the prediction against a lazy baseline.

| metric | what it measures |
|---|---|
| `cos` | mean cosine between the prediction `g(h_t)` and the target `e_t` |
| `base` | the cosine a lazy predictor gets by always outputting the batch-mean embedding |
| `skill` | `cos − base`: how much better than the lazy forecast; the number that shows real prediction |
| `erank` | effective rank of the centered embeddings: how many directions frames actually vary along. Collapse drives it toward 1 |
| `klraw` | `KL(posterior ‖ prior)` before free bits: how much information from each frame enters the recurrent state |

## Results so far

These are early, single-seed runs of 50,000 environment steps each, about 5% of the standard Crafter budget.

| run | configuration | skill at end | base | return, first → last fifth |
|---|---|---|---|---|
| A | first working version, training ratio 32, β_cdp 1 | 0.000 | 0.98 | 1.59 → 1.28 and 1.44 → 1.54 (two runs) |
| B | + DreamerV3-faithful fixes, training ratio 128, β_cdp 500 | **+0.34** | 0.58 | **1.59 → 2.51** |

![skill and baseline](docs/forecasting.png)

![effective rank and KL](docs/representation.png)

![episode return](docs/returns.png)

Run B changed several things at once. Most of them bring the code in line with DreamerV3: GroupNorm, SiLU and centered pixels in the encoder; free bits applied to the total KL rather than per latent group; rewards and continues attached to the state that observes their outcome; survival-weighted actor and critic losses; smoothed return normalization; and a world-model gradient clip of 1000. Two are genuine departures from the paper: a training ratio of 128 instead of 32, and β_cdp = 500 instead of 1.

Three bugs in run A explain most of the difference. With per-group free bits, 32 groups each got a 1-nat floor, 32 nats in total; the real KL was about 10 nats, so both KL terms sent zero gradient, and the prior, which imagination runs on, was never trained. The encoder used ReLU with no normalization, so every embedding entry was positive and all frames shared one dominant direction, which is why `base` sat at 0.98. And the reward head was asked to predict the reward of an action it had not yet seen.

Comparing at matched update counts, run B reached skill +0.065 by update 440 while run A was still at 0.000 after 2,000 updates, so the higher training ratio alone does not explain the gain. Whether the corrections are enough on their own, or whether β_cdp = 500 is needed, is the next experiment.

For scale: a return of 2.5 corresponds to two or three Crafter achievements per episode, against about 1.5 for a random agent. The agent is learning, but these numbers are not yet comparable to the paper's Crafter score, which is measured after 1 million steps.

## Next

1. Corrections only: training ratio 32 and β_cdp 1, the paper's configuration with the fixed implementation.
2. A β_cdp sweep, if the corrections alone do not reproduce run B.
3. A full 1-million-step run with several seeds, reporting the standard Crafter score.
4. Extensions under consideration: a categorical distributional energy in place of the cosine, spatial masking of the encoder input, and evaluation under structured occlusion.

## Repository layout

```text
config.py          hyperparameters: learning-rate groups, training ratio, β_cdp
encoder.py         CNN encoder, 64×64×3 frame → 6144-dim embedding
rssm.py, GRU.py    recurrent state-space model: prior, posterior, imagination step
heads.py           reward head, continue head, and the cdp predictor g(h_t)
world_model.py     world model, cdp loss, diagnostics, and a smoke test
imagine.py         imagination rollouts, λ-returns, return normalization
actor.py           policy
critic.py          distributional critic and its slow target
losses.py          actor and critic losses
replay_buffer.py   ring buffer of episodes
train.py           data collection, update budget (training ratio), logging
plot_log.py        turns a saved training log into the figures above
utils/symlog.py    symlog and two-hot encoding
```

## Running

```bash
python -m pip install torch crafter "gym==0.23.1" "numpy<2" tqdm matplotlib

python world_model.py                                    # smoke test: losses, gradients, alignment
caffeinate -i python train.py | tee runs/my_run.txt      # macOS: keep awake and save the log
python plot_log.py runs/my_run.txt:"my run" --out docs   # regenerate the figures
```

Crafter needs the old Gym API, so pin `gym==0.23.1` and `numpy<2`. On an M-series Mac, training runs at about 21 environment steps per second at a training ratio of 32 (roughly 13 hours for 1 million steps) and about 5 per second at a ratio of 128.

## References

- Hauri & Zenke (2026). *Dreamer-CDP: Improving Reconstruction-free World Models via Continuous Deterministic Representation Prediction.* arXiv:2603.07083.
- Hafner et al. (2023). *Mastering Diverse Domains through World Models* (DreamerV3).
- Tang et al. (2023). *Understanding Self-Predictive Learning for Reinforcement Learning.*
- Hafner (2021). *Benchmarking the Spectrum of Agent Capabilities* (Crafter).
