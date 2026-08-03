"""Step 6: package the agent into a Kaggle submission bundle.

    python scripts/package_submission.py

Input:   models/registry/best_policy.pt
         data/processed/static_feature_table.pt
         src/models/standalone_agent.py
Output:  submission/build/ staging directory and submission/submission.tar.gz

Bundle layout:

    submission/build/
    ├── main.py                 entrypoint: agent(obs, config), lazily
    │                           constructing a module-level agent singleton
    ├── best_model.pt           trained weights (~3 MB)
    ├── static_table.pt         precomputed (1268, 72) lookup tensor
    ├── standalone_agent.py     inlined inference code, no external imports
    └── deck.csv                the 60-card deck the agent plays

Note the deck: the bundle needs one, and the agent returns it when
obs["select"] is None (the initial selection). See the sample submission at
pokemon-tcg-ai-battle/sample_submission/ and the decks under
pokemon-tcg-ai-battle/arena/agents/*/deck.csv.

Validate before archiving — a bundle that fails on Kaggle costs a submission
slot:
  - model file size within limits;
  - inference latency < 10 ms per turn on CPU (design rule 1);
  - the bundle runs standalone: import it from a clean working directory with
    only submission/build on sys.path, and play a full match against a
    baseline. Relative imports reaching outside the folder are the classic
    failure (design rule 4).

A reference implementation of the copy/write/tar steps is in
docs/PTCG Project Architecture & Workflow Specification.md §4.
"""

if __name__ == "__main__":
    raise NotImplementedError("scripts/package_submission.py is a stub")
