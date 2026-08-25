# Garment Extraction Playground

Local Streamlit tool for testing your garment-extraction pipeline against
different models — Azure OpenAI vs. Google Gemini — with the same flow you
run in production (extract → refill missing fields → color QA → name →
description) and per-step token/cost tracking.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Usage

1. In the sidebar, configure **Model A** and (optionally) **Model B**:
   - Pick a preset (auto-fills provider + rough pricing) or go Custom.
   - Azure: paste your endpoint, API key, deployment name, API version.
   - Gemini: paste your Gemini API key and model name (e.g. `gemini-2.5-flash-lite`).
   - Edit the $/1M input/output prices if the preset is stale — pricing
     changes, always double check against the provider's current page.
2. Upload a front image (required) and back image (optional).
3. Fill in category/subcategory/color/composition/etc. — same fields your
   production `AnalysisInput` takes.
4. Click **Run Model A**, **Run Model B**, or **Run Both & Compare**.
5. Each result shows: product name, extracted attributes (JSON), SEO
   description + keywords, color QA (if a color was given), per-step token
   breakdown, and estimated cost. Running both side by side also shows a
   cost diff at the bottom.

## Notes

- **Prompts are condensed** versions of your production prompts — same
  structural rules (pocket-counting-from-correct-image, color-must-be-bare,
  don't-default-to-Solid, etc.) but shorter. If you want a tighter apples-
  to-apples accuracy comparison against your real production output, copy
  your exact prompt strings from `app/services/analysis/extractor.py` into
  the `build_*_prompt` functions near the top of `app.py`.
- **Template/SKU-lock/DB logic is intentionally left out** — this tool is
  for testing the LLM extraction quality + cost per model, not exercising
  your Mongo-backed template cache. Every run does a full extraction.
- **API keys never leave your machine** — this is a local script, nothing
  is sent anywhere except directly to Azure/Google's APIs from your own
  process.
- Gemini image support in `langchain-google-genai` expects the same
  `image_url` content-block shape used here; if you hit a compatibility
  error, check your installed `langchain-google-genai` version.
