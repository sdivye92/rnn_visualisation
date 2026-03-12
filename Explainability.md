# Explainability Methods for RNN/LSTM Inference

This document explains why the app uses these explainability methods, how each one works, and how to interpret outputs for both text generation and sentiment classification.

---

## Why explainability here?

RNN/LSTM models compress sequence context into hidden states. That gives good sequential modeling power but can hide *why* a specific token was generated or why a review was classified as positive/negative.

The goal of this explainability stack is to answer **complementary questions**:

- Which tokens had strongest influence? (magnitude)
- Did a token support or oppose the final decision? (sign)
- When did the model become confident? (timeline)
- Is influence coming from gradient sensitivity or actual directional contribution?

No single method answers all four reliably, so the app combines five methods.

---

## Method 1: Gradient Influence

### Question answered

Which input tokens is the output most sensitive to?

### Formula

For token embedding `e_t` and target score `s`:

`g_t = || d s / d e_t ||_2`

### Intuition

If tiny perturbations around token `t` cause a large output change, token `t` is important.

### Implementation notes

- One forward + one backward pass
- Fast and standard for neural attribution
- Unsiged magnitude (importance, not direction)

### Typical use

Quick ranking of important tokens.

---

## Method 2: Information Flow Heatmap

### Question answered

How does influence distribute across tokens in an immediately visual way?

### Formula

Normalize gradient influence scores:

`h_t = (g_t - min(g)) / (max(g) - min(g) + eps)`

### Intuition

A compact color map to spot concentrated vs distributed influence.

### Implementation notes

- Derived directly from Method 1
- Very interpretable for non-technical users

### Typical use

Visual scan before deeper diagnostics.

---

## Method 3: Hidden State Delta Projection (Custom, Signed)

### Question answered

Did each token push the model *toward* or *away from* the final prediction?

### Formula

Let hidden state at step `t` be `h_t`, then:

`delta_h_t = h_t - h_(t-1)`

For target output weight vector `w_target`:

`c_t = delta_h_t . w_target`

`c_t > 0` supports target, `c_t < 0` opposes target.

### Intuition

Think of each token as nudging an internal “decision pointer” in hidden-state space. This method measures each nudge along the exact decision axis used by the output layer.

### Why selected

- Signed contributions are highly intuitive (green support, red oppose)
- Mechanistic: directly tied to internal state transitions
- Works for both generation and classification

### Implementation notes

- Uses hidden states retained from forward pass
- No token masking required
- Typically one pass, very efficient

---

## Method 4: Cumulative Confidence Curve (Custom, Temporal)

### Question answered

At what point in the sequence did the model become confident?

### Formula

For each prefix `x_1:t`, compute target probability:

`p_t = softmax(f(x_1:t))[target]`

Plot `p_t` over `t`.

### Intuition

Like watching a jury confidence meter during testimony. Some words sharply increase confidence, others weaken it.

### Why selected

- Extremely intuitive timeline for humans
- Highlights turning points and uncertainty phases
- Useful for debugging failure cases

### Implementation notes

- Runs model over prefixes
- Slightly more compute than one-pass gradients, still practical

---

## Method 5: Embedding-Weighted Gradients (Input x Gradient)

### Question answered

What is each token’s directional contribution after accounting for both token value and local sensitivity?

### Formula

For embedding `e_t` and gradient `d s / d e_t`:

`ixg_t = e_t * (d s / d e_t)`

Signed scalar contribution:

`signed_t = sum(ixg_t)`

Magnitude contribution:

`mag_t = ||ixg_t||_2`

### Intuition

Raw gradients only show local slope. Input x Gradient weights slope by actual token position in embedding space, giving a stronger notion of contribution.

### Why selected

- Better directional signal than plain gradient norm
- Simple, fast, and literature-backed
- Provides both signed and magnitude views

### Implementation notes

- One forward + one backward
- Sensitive to local nonlinearity like other gradient methods

---

## Why not occlusion for token contribution?

Occlusion (mask one token, re-run, measure output drop) is easy to understand but has limits:

- `O(n)` forward passes per example (slow for long sequences)
- Can produce out-of-distribution masked inputs
- Usually gives unsigned magnitude only
- Harder to use interactively in a Streamlit UI at scale

The selected custom methods retain intuitive semantics while staying fast and interactive.

---

## Comparison (summary)

- **Gradient Influence**
  - Speed: fast
  - Signed: no
  - Strength: simple importance
  - Limitation: no direction
- **Information Flow Heatmap**
  - Speed: fast
  - Signed: no
  - Strength: visual overview
  - Limitation: derived view, not independent evidence
- **Hidden State Delta Projection**
  - Speed: fast
  - Signed: yes
  - Strength: mechanistic support/opposition signal
  - Limitation: depends on output-layer linearization axis
- **Cumulative Confidence Curve**
  - Speed: medium
  - Signed: indirectly (via curve slope)
  - Strength: temporal decision narrative
  - Limitation: prefix re-evaluation needed
- **Input x Gradient**
  - Speed: fast
  - Signed: yes (with signed aggregate)
  - Strength: directional gradient attribution
  - Limitation: still local gradient-based approximation

---

## Worked Example (conceptual)

Suppose classification input:

`"the plot starts slow but becomes brilliant"`

Predicted label: positive.

Expected interpretation:

- Gradient Influence highlights `slow`, `but`, `brilliant`
- Heatmap shows strongest region near the tail phrase
- Delta Projection:
  - `slow` negative
  - `but` transitional/slightly negative
  - `brilliant` strongly positive
- Confidence Curve:
  - low confidence through early neutral words
  - dip at `slow`
  - sharp rise at `brilliant`
- Input x Gradient:
  - high positive signed score for `brilliant`
  - negative score for `slow`

Together, this gives both **what mattered** and **how it changed the decision**.

---

## Practical interpretation guide

- If Gradient Influence and Input x Gradient both spike on a token: high confidence in token importance.
- If Delta Projection is negative but gradient magnitude is high: token is important as a counter-signal.
- If confidence curve jumps without large per-token magnitudes: effect may come from cumulative context interaction.
- For contradictory signals, inspect neighboring tokens; RNN/LSTM effects are context-coupled.

---

## References

- Simonyan, Vedaldi, Zisserman (2013), *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps*
- Sundararajan, Taly, Yan (2017), *Axiomatic Attribution for Deep Networks* (Integrated Gradients)
- Shrikumar, Greenside, Kundaje (2017), *Learning Important Features Through Propagating Activation Differences* (DeepLIFT)
- Arras et al. (2017), *Explaining Recurrent Neural Network Predictions in Sentiment Analysis* (LRP for RNNs)

