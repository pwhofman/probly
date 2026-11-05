# LLM-probly experiments

Base: pwhofman/probly @ 1189b3a

## Setup
    git clone git@github.com:elifkaracaa/probly.git
    cd probly
    uv sync
    uv add --dev ipykernel datasets scikit-learn scipy

## Structure
- common/   shared code (data.py: B1, evaluate.py + io.py: B2, reflexive.py: R1+R2)
- scripts/  generate.py (B1)
- data/qwen0.5b/      triviaqa_5.jsonl, triviaqa_50.jsonl
- results/qwen0.5b/   <method>.json
- results/sanity/     <method>_<model>.json
- notebooks/          one notebook per person + 00_summary.ipynb

## Rules
- git pull at the start of the day, git push at the end
- only edit your own notebook and results files
- changes to common/ via pull request
