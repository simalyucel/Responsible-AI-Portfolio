# AI Trust Inspector

**Can this model be trusted in production? Accuracy alone doesn't answer that.**

🔗 **[Try the live dashboard →](https://responsible-ai-portfolio-fzrvervhglcnnk5azentee.streamlit.app)**

---

## Why

Machine learning models are routinely evaluated — and deployed — on a single
number: accuracy. That number can look excellent while the model quietly
fails the group it's supposed to serve, relies on features that act as
proxies for protected characteristics, or simply can't be explained to the
people affected by its decisions. In regulated domains like education,
credit, and employment, these are exactly the risks the EU AI Act was
written to address. This project builds a small, honest tool for surfacing
that gap between "high accuracy" and "actually trustworthy."

## Goal

Audit a single ML system — not this dashboard, but the model *being*
audited — across three independent questions:

1. **Does it work?** (performance)
2. **Does it work the same for everyone?** (fairness)
3. **Why does it decide what it decides?** (explainability)

Then connect whatever the audit finds to the specific EU AI Act provisions
that anticipate that kind of problem. This is explicitly an **educational
mapping exercise, not a legal compliance assessment** — the author is not a
lawyer, and nothing here constitutes a conformity assessment under the Act.

## Methods

- **Dataset:** [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (id 320)
- **Target:** pass/fail, defined as final grade `G3 ≥ 10`
- **Deliberate exclusion:** first- and second-period grades (`G1`, `G2`) are
  dropped from the features. Keeping them in would let the model largely
  echo earlier grades back — trivially accurate, but uninformative about
  fairness, and unrealistic for the early-warning use case this system
  stands in for (a real early-warning tool has to work *before* mid-year
  grades exist).
- **Protected attribute:** `sex`
- **Model:** Logistic Regression (scikit-learn pipeline: `StandardScaler` +
  `OneHotEncoder` + `LogisticRegression`)
- **Fairness metrics:** Demographic Parity Difference, Equalized Odds
  Difference, and group-wise fail-class recall ([Fairlearn](https://fairlearn.org/))
- **Explainability:** SHAP feature importance ([SHAP](https://shap.readthedocs.io/))
- **Validation:** a single 70/30 split for the headline metrics, plus 5-fold
  stratified cross-validation to check whether the fairness gap held up
  across resampling rather than being an artifact of one split

## Dashboard

Built with [Streamlit](https://streamlit.io/). Five sections, in order:

**Model Information → Performance → Fairness → Explainability → EU AI Act Lens**

The last section renders a small, editable list of findings
(`REGULATORY_FINDINGS` in `dashboard.py`) — each one pairs a specific
measured result with the EU AI Act article or recital it maps to. Adding a
new finding is just appending a new entry to that list; nothing else in the
app needs to change.

## Results

**Performance:** overall accuracy is 81%, with precision/recall/F1 all
above 0.87 — on paper, a strong-looking model. But those numbers are
computed mostly against the majority "pass" class (~85% of students).
Fail-class recall — the ability to catch students who actually fail — is
only **~27%**. If this system exists to flag at-risk students, its real
accuracy on that task is closer to 27% than to 81%.

**Fairness:** Demographic Parity Difference (0.118) and Equalized Odds
Difference (0.304) disagree sharply, and that disagreement is the finding.
By-group accuracy looks close (female: 0.826, male: 0.788) — which would
suggest nothing is wrong. But conditioning on actual outcome tells a
different story: across 5-fold cross-validation, fail-class recall averaged
**0.375 for male students and 0.168 for female students**, and male
fail-recall exceeded female fail-recall in every single fold. The group
with the *better* overall accuracy is the one the model fails hardest at
protecting — a gap that's invisible if you only look at averages.

**Explainability:** SHAP ranks which school a student attends above the
student's own prior failures, and mother's education level (`Medu`) both
ranks in the top 6 features and carries its own Demographic Parity
Difference of 0.215. Together, these are a plausible mechanism behind the
fairness gap above: the model may be leaning on proxies for a student's
environment and socioeconomic background rather than the student's own
record.

**EU AI Act Lens:** these three findings map respectively to Article 15
(accuracy must match the system's actual purpose, not just look high on
average), Recital 56 and Article 15 together (education systems must not
reproduce discriminatory patterns), and Article 10(2) (providers must
examine training data for proxies of protected or socioeconomic
characteristics).

## Known limitations

- The dataset is small (a few hundred rows), and the fail-class subgroups
  by sex are smaller still (14–16 people each). The *direction* of the
  fairness gap held across all 5 cross-validation folds, but its exact
  *magnitude* varied considerably (roughly 1.1x to 4.8x) — worth treating
  as a real, consistent pattern rather than a precisely pinned-down number.
- Logistic Regression is intentionally simple; a more complex model might
  perform differently on both accuracy and fairness axes.
- The pass/fail threshold (`G3 ≥ 10`) is a modeling choice, not a fixed
  standard — a different threshold could shift who counts as "at risk."

## Future Improvements

- **User-uploaded datasets.** Right now the dataset and protected attribute
  are fixed. The natural next step is a `st.file_uploader` that lets a user
  bring their own CSV, pick the target and protected-attribute columns from
  dropdowns, and run the same audit pipeline against arbitrary data.
- Longer-term, out of scope for now: distribution shift / drift monitoring,
  counterfactual explanations, deeper Annex IV documentation automation,
  and comparing multiple models side by side.

## Running locally

```bash
git clone https://github.com/simalyucel/Responsible-AI-Portfolio.git
cd Responsible-AI-Portfolio/AI-Trust-Inspector
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## Disclaimer

This is an educational project built to practice fairness auditing and
explainability tooling. It is not legal advice, and the EU AI Act Lens
section is not a conformity assessment.
