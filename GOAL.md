***Description of a practical “taxonomy classification” architecture for LLMs, and the key insight:***

**“General classification” and “fine-grained classification” fail in different ways, so you shouldn’t force one mechanic to do both.**
But you can still treat it as **one system** (one taxonomy, one pipeline, one set of guardrails) by switching mechanics mid-stream.

Below is what each part means and why it works.

---

## 1) “General vs fine-grained classification are different failure modes”

### General classification (coarse / high-level)

Goal: put an input into the right *region* of a taxonomy.

Typical failure modes:

* **Recall failure**: the true label region isn’t even considered (you go to the wrong neighborhood).
* **Context explosion**: if you paste the whole taxonomy into the prompt, it’s huge, slow, and still easy to miss.
* **Semantic mismatch**: LLMs can over-index on surface cues; embeddings tend to be better at “what is this similar to?”

So general classification mainly needs:

* High recall routing
* Efficiency (don’t load everything)
* Robustness to varied phrasing

### Fine-grained classification (deep / specific leaf)

Goal: pick the *exact* label among many similar siblings (often with subtle differences).

Typical failure modes:

* **Hallucinated labels**: the model invents a plausible-sounding category that isn’t in your taxonomy.
* **Invalid jumps**: it selects a label from the wrong branch.
* **Overconfidence**: it picks something “close enough” instead of admitting ambiguity.
* **Sibling confusion**: multiple children look similar; free-form generation is slippery here.

So fine-grained classification mainly needs:

* Guaranteed validity (only real labels)
* Controlled search over siblings
* A clean abstain path when it’s ambiguous

That’s why the text says: **“slightly different mechanics but the same underlying system.”** Same taxonomy + same end goal; different tools for different stages.

---

## 2) What worked: General classification via embeddings / SetFit

### “Use embeddings / SetFit to route the input to the right high-level region”

This means: represent the input text as a vector (embedding) and find the closest matching category vectors **at a coarse level**.

* **Embeddings** are great at semantic similarity and fuzzy matching.
* **SetFit** (a lightweight sentence-transformer fine-tuning approach) can be used when you want better routing accuracy with a small labeled dataset, without training a big classifier.

### Why this “keeps recall high”

Instead of asking the LLM to pick from thousands of labels, you:

1. Use embeddings to retrieve the **top-K candidate branches** (or top-K mid-level nodes).
2. Only then do you involve the LLM for finer decisions.

This reduces the risk that the “right area” is never considered.

### “Avoids loading large taxonomies into context”

You *don’t* paste the entire taxonomy into the prompt. You only pass:

* the relevant branch (a small slice), or
* the children at the current node

This scales much better as the taxonomy grows.

---

## 3) Fine-grained classification via constrained navigation (not free generation)

### “Once you're in the right region, switch from generation to constrained navigation”

This is the big mechanical change.

Instead of prompting:

> “Which label is best? Answer with the label.”

…you prompt:

> “You are at node X. Here are valid children {A, B, C}. Choose one.”

You repeat this until you reach a leaf (or decide to stop).

### “At each step, the model can only choose from valid children”

This is basically **tree traversal** with a restricted action space:

* Current node → list children → pick one child → move there → repeat

### “Invalid labels simply cannot be produced”

Because the model is not allowed to output arbitrary text. You enforce a strict output schema like:

* `{"choice_id": 2}`
  or a function/tool call
  or a logit-bias / constrained decoding approach
  or server-side validation that rejects anything not in the list

So hallucinated categories disappear by construction.

---

## 4) “So the pipeline becomes…”

> **semantic recall → constrained traversal → optional sibling contrast → abstain if ambiguous**

Here’s what each stage means.

### A) Semantic recall (routing)

Use embeddings to retrieve likely regions:

* top-K branches
* or top-K internal nodes
  This gives you candidates with high recall.

### B) Constrained traversal (stepwise navigation)

Within the selected branch:

* Start at a node
* Present only its children
* Force the model to pick among them
* Continue until a stopping rule triggers (leaf / confidence / depth limit / ambiguity)

This turns classification into a **controlled search problem**, not a creative generation problem.

### C) Optional sibling contrast

When two siblings are close, do a deliberate comparison:

* “Compare A vs B. What evidence supports each? What would disprove each?”
* Or ask for a short rationale tied to the input text
* Or run a small local scoring step across siblings

This is a “precision booster” for confusing sibling sets.

### D) Abstain if ambiguous

Instead of guessing, the system can output:

* `ABSTAIN`
* or “needs human review”
* or “top 2 candidates with uncertainty”
* or “request clarification”

This is crucial in real taxonomies where labels overlap.

---

## 5) “No retries. No hallucinated categories.”

### No retries

The idea is: if the pipeline is designed correctly, you shouldn’t need “try again” loops.

* Retrieval gives you candidates
* Traversal guarantees valid outputs
* Abstain handles uncertainty

Retries often mask structural issues and can introduce inconsistency.

### No hallucinated categories

Constrained traversal + strict validation means:

* Only known labels can be selected
* The model can’t invent new ones

---

## 6) “Embeddings do what they're good at… Constraints do what they're good at…”

This is the core division of labor:

* **Embeddings / SetFit**: *Similarity, routing, recall, scale*
* **Constraints / traversal**: *Validity, precision, determinism, safety*

Together you get:

* scalable taxonomy handling (even huge)
* robust accuracy
* low hallucination risk
* easier maintenance when taxonomy changes

---

## 7) “Scales across flat, hierarchical, and fast-changing taxonomies”

### Flat taxonomies

Even if labels are “flat,” you can simulate hierarchy by:

* clustering labels into regions
* creating “virtual parents” (groups)
* routing → pick group → pick label

### Hierarchical taxonomies

Traversal is a natural fit: you literally walk the tree.

### Fast-changing taxonomies

Because you are not “teaching the model the whole taxonomy in prompt,” and not relying purely on memorized label strings:

* you can update the label store
* re-embed labels
* the routing/traversal still works

Less overfitting to one query type because the system is modular:

* retrieval handles broad semantic variety
* traversal handles structural correctness

---

## A concrete mini-example

Imagine a taxonomy:

* **Payments**

  * Chargebacks
  * Refunds
  * Disputes
* **Account Access**

  * Password reset
  * MFA issues
  * Locked account

Input: “Customer says their MFA code never arrives and they’re locked out.”

1. **Semantic recall (embeddings)** → top region: “Account Access”
2. **Traversal** at “Account Access” children = {Password reset, MFA issues, Locked account}
   Model picks “MFA issues” (maybe with “Locked account” as close sibling)
3. **Sibling contrast** between MFA issues vs Locked account
4. If truly unclear, **abstain** or ask: “Are they locked out due to too many failed attempts, or solely missing MFA codes?”

And at no point can it output “Two-factor delivery failure category v2” (hallucination), because it’s not a valid child.

---
