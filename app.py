"""
Garment Extraction Playground
==============================

A local Streamlit harness for trying your garment-extraction LLM pipeline
against different models/providers (Azure OpenAI vs. Google Gemini) with
side-by-side accuracy + cost comparison.

Mirrors the flow in your production `GarmentExtractor._run_full_extraction`:
  1. extract_initial_attributes  (vision -> structured schema)
  2. refill_missing_with_llm     (fills any blank fields)
  3. derive_number_of_pockets    (code-computed, never trust the model's sum)
  4. verify_color_match          (QA pass, only if you supply a color)
  5. generate_product_name       (plain LLM call)
  6. generate_product_description(plain LLM call, paragraph + keywords)

Every LLM call is instrumented with a TokenTracker so you get the same
per-step token/cost breakdown you have in production, but with prices you
can edit live in the sidebar (since pricing differs per provider/model and
changes over time).

NOTE: The extraction prompts here are condensed versions of your production
prompts — same structural intent (pocket counting rules, color-bareness
rule, pattern-vs-solid rule, etc.) but shorter, to keep this file readable.
Edit PROMPT_* constants near the top if you want tighter parity with your
exact production wording before trusting comparative accuracy results.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

# ======================================================================
# Schema — a standalone equivalent of your GarmentSchema/GarmentAnalysis
# ======================================================================


class GarmentAnalysis(BaseModel):
    garment_type: Optional[str] = None
    garment_category: Optional[str] = None
    color: Optional[str] = None
    fit: Optional[str] = None
    length: Optional[str] = None
    collar: Optional[str] = None
    neckline: Optional[str] = None
    neckline_details: Optional[str] = None
    sleeves_type: Optional[str] = None
    sleeve_styling: Optional[str] = None
    waist_type: Optional[str] = None
    leg_style: Optional[str] = None
    hem_finish: Optional[str] = None
    pattern_type: Optional[str] = None
    front_pocket_count: Optional[int] = 0
    back_pocket_count: Optional[int] = 0
    number_of_pockets: Optional[str] = None
    pocket_description: Optional[str] = None
    design_details: Optional[str] = None
    back_design_details: Optional[str] = None
    material_composition: Optional[str] = None
    brand_color_code: Optional[str] = None
    season: Optional[str] = None
    ideal_for: Optional[str] = None
    style: Optional[str] = None
    tag_size: Optional[str] = None
    tag_color_code: Optional[str] = None


class GarmentSchema(BaseModel):
    garment_analysis: GarmentAnalysis = Field(default_factory=GarmentAnalysis)
    ean_code: Optional[str] = None


CAPITALIZE_FIELDS = [
    "color", "fit", "length", "collar", "neckline_details",
    "sleeve_styling", "waist_type", "leg_style", "hem_finish", "pattern_type",
]

# ======================================================================
# Token / cost tracking — same shape as your production TokenTracker,
# but prices are injected per-run instead of hardcoded, since you'll be
# swapping models constantly in this tool.
# ======================================================================


class TokenTracker:
    def __init__(self, price_input_per_m: float, price_output_per_m: float):
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.steps: List[Tuple[str, int, int]] = []

    def record(self, label: str, ai_message) -> None:
        if ai_message is None:
            return
        usage = getattr(ai_message, "usage_metadata", None) or {}
        if not usage:
            token_usage = (getattr(ai_message, "response_metadata", None) or {}).get("token_usage", {}) or {}
            usage = {
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
            }
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        self.steps.append((label, input_tokens, output_tokens))

    def summary(self) -> Dict[str, Any]:
        total_input = sum(s[1] for s in self.steps)
        total_output = sum(s[2] for s in self.steps)
        cost = (
            (total_input / 1_000_000) * self.price_input_per_m
            + (total_output / 1_000_000) * self.price_output_per_m
        )
        return {
            "steps": [{"label": l, "input_tokens": i, "output_tokens": o} for l, i, o in self.steps],
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "estimated_cost_usd": round(cost, 6),
        }


# ======================================================================
# Provider presets — editable starting points, not gospel. Prices drift;
# always sanity check against the provider's pricing page before trusting
# a cost comparison for a real decision.
# ======================================================================

PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "Azure OpenAI — gpt-4o-mini":        {"provider": "azure",  "model": "gpt-4o-mini",             "input": 0.15, "output": 0.60},
    "Azure OpenAI — gpt-4.1-nano":       {"provider": "azure",  "model": "gpt-4.1-nano",             "input": 0.10, "output": 0.40},
    "Azure OpenAI — gpt-4.1-mini":       {"provider": "azure",  "model": "gpt-4.1-mini",             "input": 0.40, "output": 1.60},
    "Azure OpenAI — gpt-4.1":            {"provider": "azure",  "model": "gpt-4.1",                  "input": 2.00, "output": 8.00},
    "Gemini — gemini-2.5-flash-lite":    {"provider": "gemini", "model": "gemini-2.5-flash-lite",    "input": 0.10, "output": 0.40},
    "Gemini — gemini-2.5-flash":         {"provider": "gemini", "model": "gemini-2.5-flash",         "input": 0.30, "output": 2.50},
    "Gemini — gemini-3-flash-preview":   {"provider": "gemini", "model": "gemini-3-flash-preview",   "input": 0.25, "output": 1.50},
    "Custom":                            {"provider": "azure",  "model": "",                          "input": 0.00, "output": 0.00},
}


# ======================================================================
# LLM client builders
# ======================================================================


def get_azure_llm(endpoint: str, api_key: str, deployment: str, api_version: str, model_name: str):
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment,
        api_version=api_version,
        temperature=0.0,
        model=model_name or deployment,
    )


def get_gemini_llm(api_key: str, model_name: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,
    )


def build_llm(cfg: Dict[str, Any]):
    if cfg["provider"] == "azure":
        return get_azure_llm(
            endpoint=cfg["azure_endpoint"],
            api_key=cfg["azure_api_key"],
            deployment=cfg["azure_deployment"],
            api_version=cfg["azure_api_version"],
            model_name=cfg["model"],
        )
    elif cfg["provider"] == "gemini":
        return get_gemini_llm(api_key=cfg["gemini_api_key"], model_name=cfg["model"])
    raise ValueError(f"Unknown provider: {cfg['provider']}")


# ======================================================================
# Image helpers
# ======================================================================


def file_to_data_uri(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    raw = uploaded_file.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = uploaded_file.type or "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ======================================================================
# Prompts (condensed — see module docstring)
# ======================================================================


def build_extraction_prompt(category: str, subcategory: str, has_back: bool) -> str:
    if has_back:
        pocket_instr = (
                    "Both a front image and a back image are provided (in that order). "
                    "Count front_pocket_count strictly from the "
                    "front image (and side edges, if visible there). Count "
                    "back_pocket_count strictly from the BACK image — do not estimate or "
                    "guess this from garment_type; the back image is provided specifically "
                    "so you can count it directly. Look carefully at the full back image "
                    "before settling on a number — small patch pockets can be easy to "
                    "miss. If a coin/watch pocket is visible in either image, note it in "
                    "pocket_description (see below) but do not fold it into "
                    "front_pocket_count."
                )
        pocket_description_instr = (
                    "For pocket_description: write ONE combined sentence describing every "
                    "pocket actually visible on the garment — front and back a coin/"
                    "watch pocket if present (small inset pocket, usually within or beside a "
                    "front pocket). Source each location from the SAME image you used for its "
                    "count: front pocket detail comes from the front image; back pocket "
                    "detail comes strictly from the BACK image .Example (front "
                    "+ back image provided): '2 slant front pockets with topstitching, 2 back "
                    "patch pockets with a button flap, and a coin pocket at the right hip.' "
                    "Use 'No pockets.' only if none are visible anywhere. Every location and "
                    "count mentioned in this sentence MUST exactly match whatever count "
                    "fields you report elsewhere on this schema — do not describe a pocket "
                    "that isn't visible"
                )
    else:
        pocket_instr = (
            "Only a front image is provided. Determine front_pocket_count from it. Prefer 0 "
            "over overestimating back_pocket_count since no back view exists."
        )

    classification_instr = ""
    if category or subcategory:
        classification_instr = (
                    f"This product's catalog CATEGORY is '{category}' and SUBCATEGORY is "
                    f"'{subcategory}'. Treat this as authoritative ground truth: keep "
                    f"garment_category, garment_type, and product_description consistent with "
                    f"this given subcategory. Do NOT rename, reclassify, or substitute a "
                    f"different (even if visually plausible) garment type than what the given "
                    f"subcategory indicates — for example, if the subcategory says the item is "
                    f"a generic '{subcategory}', do not narrow or rename it to a more specific "
                    f"but different term in your output."
                )
    neckline_instruction = (
                "For collar/neckline fields: look closely at the actual neckline construction "
                "before naming it — do not default to a generic guess. Distinguish clearly between "
                "e.g. round neck vs crew neck vs V-neck vs mandarin/band collar vs polo collar vs "
                "shirt collar vs boat neck vs sweetheart vs halter vs keyhole. If a collar is genuinely "
                "not applicable (e.g. bottomwear), leave it unset rather than guessing. Only report a "
                "neckline detail you can actually see in the image — do not infer one from garment_type alone."
            )

    sleeve_instruction = (
            "sleeves_type and sleeve_styling are DIFFERENT fields — do not conflate them. "
            "sleeves_type describes sleeve LENGTH/COVERAGE (full sleeve, half sleeve, sleeveless, "
            "3/4th sleeve, cap sleeve). sleeve_styling describes the CONSTRUCTION/SILHOUETTE of the "
            "sleeve independent of length (Regular, Puff, Raglan, Cuffed, Bell, Balloon). Extract both "
            "independently from what's visible — do not copy one value into the other."
        )

    design_details_instruction = (
        "For design_details: list ONLY structural/construction design elements — "
        "embroidery type and placement, prints, panel/yoke construction, trims, "
        "tassels, mirror work, gathers, pleats, cutouts, appliqué, lace, piping. "
        "Do NOT mention color anywhere in this field, including the base garment "
        "color or any secondary/trim color — color belongs strictly in the color "
        "field, never here. Write 2-4 short, concrete, catalog-style sentences, "
        "not a flowing product-description paragraph. If no notable design elements "
        "are visible beyond the base construction, write 'No additional design "
        "embellishments.'"
    )

    ideal_for_instruction = (
        "For ideal_for: derive the occasion/use-case for THIS specific garment by "
        "reasoning from what's actually visible — do not select from any fixed "
        "phrase and do not default to 'casual outings' or 'relaxing at home' unless "
        "the garment genuinely signals that level of informality (e.g. simple jersey "
        "cotton with no print/embellishment). The true range of occasions spans far "
        "wider than casual/homewear — consider the full spectrum: loungewear/"
        "homewear, everyday errands, casual day-to-day wear, workwear/office wear, "
        "semi-formal day events, festive or celebration wear, wedding-guest wear, "
        "religious/traditional functions, evening/party wear — and pick based on "
        "actual visual evidence, not habit. Weigh: fabric (jersey/basic cotton = "
        "everyday; structured cotton or linen with a print = smart-casual day wear; "
        "silk/velvet/net/heavily worked fabric = festive or evening); embellishment "
        "(none = everyday, moderate print/embroidery = smart-casual or semi-formal, "
        "heavy embroidery/sequins/zari = festive/celebration); silhouette (relaxed/"
        f"loose = comfort-first, tailored/structured = more occasion-appropriate); "
        f"and print/color formality. Base it on THIS garment's own visible signals "
        f"and the given subcategory ('{subcategory}') — a printed, structured, "
        "3/4-sleeve dress or kurta is NOT automatically homewear just because it's "
        "ethnic or has a relaxed fit; assess it the same way you would any other "
        "garment on its actual visible formality."
    )


    color_instruction = (
        "For color: report the single most visually dominant/prominent color of the garment as "
        "a common, catalog-friendly color name (e.g. 'Cream', 'Navy Blue', 'Dusty Pink') — not a "
        "vague or overly specific paint-swatch name. If the garment has a secondary trim/print "
        "color, that belongs in design_details, not in color. Do NOT prepend any "
        "qualifier or adjective — such as 'dark', 'light', 'bright', 'solid', or 'plain' — "
        "directly next to the color mention, unless that qualifier is genuinely part of the "
        "color's own name (e.g. 'Dark Grey' as a distinct named shade, not 'dark brown' as a "
        "description of a brown item, and not 'solid brown' either). Whenever design_details "
        "mentions the color, it must appear as a bare, standalone mention — the exact same "
        "word(s) as the color field, with nothing appended directly before or after it. This "
        "keeps every color mention consistent and maximally accurate across fields."
    )

    barcode_instruction = (
        "For ean_code/barcode_id: only report digits you can actually read printed under or near "
        "a barcode/tag in the image. Do not guess, round, or fabricate a barcode number — if it "
        "is not legibly visible, leave it unset."
    )

    pattern_instruction = (
        "For pattern_type: look closely at the actual print/pattern construction "
        "visible on the garment before naming it. Before ever writing 'Solid', "
        "explicitly check: is there ANY repeating motif, print, weave variation, or "
        "color variation visible across the fabric? If yes, it is NOT solid — name "
        "the real pattern specifically (e.g. Ikat, Bandhani, Block Print, Ajrakh, "
        "Kalamkari, Zari Woven, Booti, Paisley, Batik, Floral, Damask, Geometric, "
        "Tie-Dye, Abstract). This list is illustrative, not exhaustive — if the true "
        "pattern isn't listed here, name it correctly anyway. A wavy, feathered, or "
        "medallion-style repeating motif (light-on-dark or dark-on-light) is "
        "typically an Ikat or Block Print, NOT Solid — do not mislabel a printed "
        "fabric as Solid just because it uses only 1-2 colors. Reserve 'Solid' "
        "strictly for a genuinely flat, single-color fabric with zero print, weave "
        "pattern, or motif variation anywhere on the garment."
    )

    back_detail_instruction = (
        "For back_design_details: "
        "Describe what's specifically visible on the BACK — print/pattern "
        "continuation, back yoke, zipper/button closure, back neckline shape, "
        "cutouts, back pockets. If the back has no distinct construction of its own "
        "and simply continues the same fabric as the front, describe it accurately "
        "based on what that front fabric actually is — name the real print/pattern "
        "(e.g. 'The ikat print carries through consistently from front to back.', "
        "'The floral print continues uninterrupted onto the back.') rather than "
        "defaulting to generic phrasing like 'uniform solid look' or 'clean back' — "
        "those words are WRONG and must not be used unless the garment is genuinely "
        "a plain solid color with no print. if it is plain solid with no print say that it no print but in a better vocabulary like in reference to the front side print an example can be the the front side has floral prints but the backside is clean or solid with no prints"
    )  
    
    style_instruction = (
            "For style: assign a short catalog silhouette label appropriate to the "
            "garment_type — e.g. 'Straight', 'Wide Leg', 'Slim', 'Ankle', 'Trouser' for "
            "bottomwear; 'T-Shirt Style', 'Polo', 'Regular Fit' for topwear; 'Wrap', "
            "'Kurta Set' for dresses/sets; 'Bralette', 'Brief' for innerwear. Base it only "
            "on visible silhouette/construction, not guesswork."
        )
    
    return f"""Analyze the image(s) and fill the structured schema with maximum accuracy. 
                If a garment is visible, extract all garment details. 
                If no garment is visible (only barcode, tag, or background), 
                only extract the barcode_id and skip garment attributes. 
                IMPORTANT: If a garment tag/label is visible with printed text (size, 
                fit, color code, article number, barcode, product name), treat that 
                printed text as ground truth and prioritize it over visual inference 
                for the corresponding fields — populate tag_size and tag_color_code 
                from the exact printed values when present, and set fit to the printed 
                fit label if one is shown. 
                {pocket_instr} 
                {pocket_description_instr} 
                {neckline_instruction} 
                {sleeve_instruction} 
                {style_instruction} 
                {back_detail_instruction} 
                {color_instruction} 
                {barcode_instruction} 
                {classification_instr}
                {design_details_instruction} 
                {ideal_for_instruction} 
                {pattern_instruction} 
                Base every field strictly on what is visible in the image(s) — if a detail 
                genuinely cannot be determined, leave it unset rather than guessing."""



def build_color_verify_prompt(given_color: str) -> str:
    return f"""
You are a QA specialist checking a product listing's color field against
the actual product photo.

The listing claims the garment color is: {given_color}

Look closely at the garment in the image(s) and determine its single most
dominant, visually accurate color as a common, catalog-friendly color name
(e.g. 'Cream', 'Navy Blue', 'Dusty Pink') — report what you actually observe
in the image, do not just restate the claimed color.

Then compare it to the claimed color and decide whether they refer to the
same color. Minor phrasing/formatting differences (e.g. 'Off White' vs
'Off-White') still count as a match; genuinely different colors or shades
do not.

OUTPUT FORMAT (STRICT, valid JSON only, no preamble or markdown fences):
{{
  "ai_color": "<color you observe in the image>",
  "color_flag": "Match" or "Mismatch"
}}
"""


def build_product_name_prompt(sku_id: str, garment_dict: dict, category: str, subcategory: str, color: str, composition: str) -> str:

    garment_analysis = garment_dict.get("garment_analysis", garment_dict) or {}
    color = color or garment_analysis.get("color") or ""
    material_composition = composition or garment_analysis.get("material_composition") or ""

    return f"""
        You are an expert e-commerce catalog specialist.

        CATEGORY: {category}
        SUBCATEGORY: {subcategory}

        EXTRACTED GARMENT ATTRIBUTES (GROUND TRUTH):
        {garment_dict}

        COLOR (must appear in name): {color}
        MATERIAL COMPOSITION (primary fabric must appear in name): {material_composition}

        TASK:
        1. Generate ONE clean, professional product name.
        2. Do NOT include any SKU, product ID, or numeric code in the name.
        3. The SUBCATEGORY value given above ("{subcategory}") MUST be used verbatim (or with
           only minor grammatical adjustment, e.g. pluralization) as the garment-type term in
           the name. Do NOT substitute it with a different or more specific garment word — for
           example, if SUBCATEGORY is "lower", the name must say "Lower", NOT "Trousers",
           "Pants", "Jeans", or any other synonym, even if the image looks like that garment.
        4. Combine category, the subcategory term (as-is), and key attributes:
           - color (MANDATORY — always include, using the exact value given above)
           - primary fabric derived from material_composition (MANDATORY — always include,
             e.g. if material_composition is "95% Cotton 5% Elastane", use "Cotton"; if
             "100% Linen", use "Linen")
           - fit
           - sleeves (if relevant, topwear only)
        5. Do NOT invent attributes not present in the extracted data above. Do NOT state a
           color or fabric in the name that differs from the COLOR and MATERIAL COMPOSITION
           given above. Use the COLOR value exactly as given — do not substitute a synonym
           or a different shade name for it, and do not attach an adjective directly next to
           it either (no 'Solid <color>', no 'Dark <color>', no 'Plain <color>', etc.) — it
           must appear as a bare, standalone word.
        6. Avoid marketing words like "premium", "best", "trendy".
        7. Keep it concise and catalog-friendly, e.g.:
           "Women's Dusty Pink Cotton Relaxed Fit Lower"
           (note: color "Dusty Pink" and fabric "Cotton" are both present, derived from the
           COLOR and MATERIAL COMPOSITION fields — never omit either.)

        OUTPUT FORMAT (STRICT):
        {{
          "product_name": "<product name, no SKU>"
        }}
        """


def build_product_description_prompt(garment_dict: dict, category: str, subcategory: str, brand_color: str, color: str, composition: str) -> str:

    garment_analysis = garment_dict.get("garment_analysis", garment_dict) or {}

    ground_truth_block = f"""
    AUTHORITATIVE PRODUCT FACTS (these override anything conflicting in the extracted attributes):
    - Garment type/subcategory to use in all text: "{subcategory}" (do NOT substitute a different
        garment noun like 'trousers', 'pants', 'skirt', etc. even if it seems more descriptive)
    - Color: {color or garment_analysis.get('color')}
    - Brand color code: {brand_color or garment_analysis.get('brand_color_code')}
    - Material composition: {composition or garment_analysis.get('material_composition')}
    """

    return f"""
You are an expert e-commerce fashion copywriter and SEO specialist.

        CATEGORY: {category}
        SUBCATEGORY: {subcategory}

        {ground_truth_block}

        EXTRACTED GARMENT ATTRIBUTES (GROUND TRUTH, but subcategory/color/composition above take priority):
        {garment_dict}

        TASK:
        1. Write ONE SEO-friendly product description paragraph, in this order (flowing
           prose, no labels):
             a) DESIGN OPENING (1-2 sentences) using design_details, but referring to the
                garment ONLY as "{subcategory}" — never a more specific synonym.
             b) BACK DETAIL (only if back_design_details is present in the extracted
                attributes above): ONE short clause folded naturally into the design
                opening or specifications sentence — e.g. "...while the clean back mirrors
                the front for a polished finish" or "...with a back yoke that continues the
                print." If back_design_details is 'Back mirrors the front.', use wording
                close to that. If back_design_details is absent/unset, do not mention the
                back at all — do not invent one.
             c) SPECIFICATIONS (1-2 sentences): use the composition and color given above
                verbatim where natural, plus fit and functional details (pockets, finish, hemline).
             d) ONE sentence starting with "Ideal for ..."
             e) ONE sentence starting with "Pair with ..."
        2. Write 15-20 long-tail "Generic Keywords" — every keyword must use "{subcategory}"
           as the garment noun, and reflect the given color/composition where relevant.
        3. Write 3-6 "Short Keywords" using "{subcategory}" as the garment noun.
        4. Do NOT invent details beyond what's given — every claim (fabric, fit, pockets,
           finish, occasion) must trace back to a field in the extracted attributes or the
           authoritative facts above. If a detail isn't present in either, omit it rather
           than inventing something plausible-sounding.
        5. Use the exact color value given above consistently everywhere the color is
           mentioned — in the paragraph, the generic keywords, and the short keywords. Do
           not substitute a synonym or a different shade name for it, and do not attach an
           extra adjective directly next to it either (no 'solid <color>', no 'dark <color>',
           no 'plain <color>', no 'bright <color>', etc.) — the color must appear as a bare,
           standalone word exactly as given, every single time it's mentioned.
        6. Avoid marketing fluff and avoid repeating the same keyword phrase twice.

        OUTPUT FORMAT (STRICT, valid JSON only):
        {{
          "paragraph": "<single paragraph>",
          "generic_keywords": ["...", "..."],
          "short_keywords": ["...", "..."]
        }}""".strip()


# ======================================================================
# Parsing / derived-field helpers (same logic as production)
# ======================================================================

import re
import json
from typing import Any

def _extract_text(content: Any) -> str:
    """Normalize LangChain message content to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # only pull actual text blocks; skip 'extras', signatures, tool_use, etc.
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
        return "\n".join(parts)
    # fallback for anything else (e.g. a message object itself)
    return str(content)


def parse_llm_json(content: Any) -> dict:
    text = _extract_text(content)
    if not text:
        return {}

    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {}

# def parse_llm_json(content: str) -> dict:
#     if not content:
#         return {}
#     cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
#     cleaned = re.sub(r"\s*```$", "", cleaned.strip())
#     try:
#         return json.loads(cleaned)
#     except json.JSONDecodeError:
#         pass
#     match = re.search(r"\{.*\}", cleaned, re.DOTALL)
#     if match:
#         candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
#         try:
#             return json.loads(candidate)
#         except json.JSONDecodeError:
#             pass
#     return {}


def capitalize_garment_dict(garment_dict: dict) -> dict:
    ga = garment_dict.get("garment_analysis", {})
    for field in CAPITALIZE_FIELDS:
        v = ga.get(field)
        if v and isinstance(v, str):
            ga[field] = v.title()
    if ga.get("brand_color_code"):
        ga["brand_color_code"] = str(ga["brand_color_code"]).upper()
    return garment_dict


def fallback_pocket_description(ga: dict) -> str:
    def phrase(count, label):
        if not count:
            return None
        noun = label if count == 1 else f"{label}s"
        return f"{count} {noun}"

    parts = [p for p in [
        phrase(ga.get("front_pocket_count") or 0, "front pocket"),
        phrase(ga.get("back_pocket_count") or 0, "back pocket"),
    ] if p]
    if not parts:
        return "No pockets."
    desc = (parts[0] if len(parts) == 1 else " and ".join(parts)) + "."
    return desc[0].upper() + desc[1:]


def derive_number_of_pockets(ga: dict) -> dict:
    total = sum(ga.get(f) or 0 for f in ("front_pocket_count", "back_pocket_count"))
    ga["number_of_pockets"] = "No Pocket" if total == 0 else str(total)
    if not ga.get("pocket_description"):
        ga["pocket_description"] = fallback_pocket_description(ga)
    return ga


# ======================================================================
# Pipeline steps (mirrors GarmentExtractor)
# ======================================================================


def extract_initial_attributes(llm, front_uri, back_uri, category, subcategory, tracker) -> GarmentSchema:
    structured_llm = llm.with_structured_output(GarmentSchema, include_raw=True)
    prompt = build_extraction_prompt(category, subcategory, bool(back_uri))
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": front_uri}},
    ]
    if back_uri:
        content.append({"type": "image_url", "image_url": {"url": back_uri}})
    response = structured_llm.invoke([HumanMessage(content=content)])
    tracker.record("initial_extraction", response.get("raw"))
    parsed = response.get("parsed")
    if parsed is None:
        raise RuntimeError(f"Initial extraction returned no parseable schema: {response.get('parsing_error')}")
    return parsed


def refill_missing_with_llm(llm, garment: GarmentSchema, front_uri, tracker) -> GarmentSchema:
    structured_llm = llm.with_structured_output(GarmentSchema, include_raw=True)
    data = garment.garment_analysis
    missing = [k for k, v in data.model_dump().items() if not v]
    if not missing:
        return garment
    prompt = f"The following garment fields are missing or undetermined:\n{missing}\n\nUsing the image, fill these fields as accurately as possible."
    content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": front_uri}}]
    response = structured_llm.invoke([HumanMessage(content=content)])
    tracker.record("refill_missing", response.get("raw"))
    update = response.get("parsed")
    if update is None:
        return garment
    update_data = update.garment_analysis
    for k, v in update_data.model_dump().items():
        if hasattr(data, k) and not getattr(data, k):
            setattr(data, k, v)
    return garment


def verify_color_match(llm, front_uri, given_color, back_uri, tracker) -> Tuple[Optional[str], Optional[str]]:
    try:
        prompt = build_color_verify_prompt(given_color)
        content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": front_uri}}]
        if back_uri:
            content.append({"type": "image_url", "image_url": {"url": back_uri}})
        response = llm.invoke([HumanMessage(content=content)])
        print(response)
        tracker.record("color_verify", response)
        parsed = parse_llm_json(response.content)
        print(parsed)
        return parsed.get("ai_color"), parsed.get("color_flag")
    except Exception as e:
        return None, f"Error: {e}"


def generate_product_name(llm, sku_id, garment_dict, category, subcategory, color, composition, tracker) -> str:
    prompt = build_product_name_prompt(sku_id, garment_dict, category, subcategory, color, composition)
    response = llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}])])
    print("generated response in product name")
    print(response)
    tracker.record("product_name", response)
    print("after tracker in product name")
    parsed = parse_llm_json(response.content)
    print(parsed)
    print("Done with parsing in product name")
    name = parsed.get("product_name")
    if not name:
        m = re.search(r'"product_name"\s*:\s*"([^"]+)"', response.content)
        name = m.group(1) if m else response.content.strip()
        print("returning results from product name")
    return name


def generate_product_description(llm, garment_dict, category, subcategory, brand_color, color, composition, tracker) -> dict:
    prompt = build_product_description_prompt(garment_dict, category, subcategory, brand_color, color, composition)
    response = llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}])])
    tracker.record("product_description", response)
    parsed = parse_llm_json(response.content)
    if not parsed:
        return {"paragraph": response.content.strip(), "generic_keywords": [], "short_keywords": []}
    parsed.setdefault("generic_keywords", [])
    parsed.setdefault("short_keywords", [])
    return parsed


def run_pipeline(cfg: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    llm = build_llm(cfg)
    tracker = TokenTracker(cfg["input_price"], cfg["output_price"])

    front_uri, back_uri = inputs["front_uri"], inputs["back_uri"]
    category, subcategory = inputs["category"], inputs["subcategory"]

    garment = extract_initial_attributes(llm, front_uri, back_uri, category, subcategory, tracker)
    print("extracted initial attributes")
    garment = refill_missing_with_llm(llm, garment, front_uri, tracker)
    print("refilled initial attributes")
    garment_dict = garment.model_dump()
    garment_dict["garment_analysis"] = derive_number_of_pockets(garment_dict["garment_analysis"])

    ga = garment_dict["garment_analysis"]
    if inputs["composition"]:
        ga["material_composition"] = inputs["composition"]
    if inputs["brand_color"]:
        ga["brand_color_code"] = inputs["brand_color"]

    ai_color, color_flag = (None, None)
    if inputs["color"]:
        ai_color, color_flag = verify_color_match(llm, front_uri, inputs["color"], back_uri, tracker)

    print("verified color")
    garment_dict = capitalize_garment_dict(garment_dict)
    ga = garment_dict["garment_analysis"]

    prominent_color = inputs["color"] or ga.get("color")
    composition = inputs["composition"] or ga.get("material_composition")

    product_name = generate_product_name(
        llm, inputs["sku_id"], garment_dict, category, subcategory, prominent_color, composition, tracker
    )
    print("generated product name")
    seo_content = generate_product_description(
        llm, garment_dict, category, subcategory, inputs["brand_color"], prominent_color, composition, tracker
    )
    print("generated product description")
    neckline = ga.get("neckline") or ga.get("neckline_details")
    ean_code = inputs["ean_code"] or inputs["sku_id"] or "N/A"

    result = {
        "sku_id": inputs["sku_id"] or "N/A",
        "product_name": product_name,
        "category": category,
        "subcategory": subcategory,
        "extracted_attributes": {"garment_analysis": ga},
        "product_description": seo_content,
        "season": ga.get("season") or inputs["season"],
        "ean_code": ean_code,
        "neckline": neckline,
        "prominent_color": prominent_color,
        "fit": ga.get("fit"),
        "sleeve_styling": ga.get("sleeve_styling"),
        "sleeve_type": ga.get("sleeves_type"),
        "ai_color": ai_color,
        "color_flag": color_flag,
    }
    return {"result": result, "token_summary": tracker.summary()}


# ======================================================================
# Streamlit UI
# ======================================================================

st.set_page_config(page_title="Garment Extraction Playground", layout="wide")
st.title("🧵 Garment Extraction Playground")
st.caption(
    "Local harness for testing your garment-extraction pipeline against different "
    "models/providers, with per-step token + cost tracking."
)


def provider_config_form(label: str, key_prefix: str) -> Dict[str, Any]:
    st.subheader(label)
    preset_name = st.selectbox("Preset", list(PROVIDER_PRESETS.keys()), key=f"{key_prefix}_preset")
    preset = PROVIDER_PRESETS[preset_name]

    provider = st.radio(
        "Provider", ["azure", "gemini"], horizontal=True,
        index=0 if preset["provider"] == "azure" else 1,
        key=f"{key_prefix}_provider",
    )

    model_name = st.text_input("Model / deployment name", value=preset["model"], key=f"{key_prefix}_model")

    cfg: Dict[str, Any] = {"provider": provider, "model": model_name}

    if provider == "azure":
        cfg["azure_endpoint"] = st.text_input("Azure endpoint", placeholder="https://<resource>.openai.azure.com/", key=f"{key_prefix}_azure_endpoint")
        cfg["azure_api_key"] = st.text_input("Azure API key", type="password", key=f"{key_prefix}_azure_key")
        cfg["azure_deployment"] = st.text_input("Deployment name", value=model_name, key=f"{key_prefix}_azure_deployment")
        cfg["azure_api_version"] = st.text_input("API version", value="2024-08-01-preview", key=f"{key_prefix}_azure_version")
    else:
        cfg["gemini_api_key"] = st.text_input("Gemini API key", type="password", key=f"{key_prefix}_gemini_key")

    col1, col2 = st.columns(2)
    cfg["input_price"] = col1.number_input("Input $/1M tokens", value=float(preset["input"]), step=0.01, key=f"{key_prefix}_price_in")
    cfg["output_price"] = col2.number_input("Output $/1M tokens", value=float(preset["output"]), step=0.01, key=f"{key_prefix}_price_out")

    return cfg


with st.sidebar:
    st.header("Model configuration")
    tab_a, tab_b = st.tabs(["Model A", "Model B"])
    with tab_a:
        cfg_a = provider_config_form("Model A", "a")
    with tab_b:
        cfg_b = provider_config_form("Model B", "b")

st.header("1. Product inputs")
c1, c2 = st.columns(2)
with c1:
    front_file = st.file_uploader("Front image (required)", type=["jpg", "jpeg", "png", "webp"])
    category = st.text_input("Category", placeholder="e.g. women")
    subcategory = st.text_input("Subcategory", placeholder="e.g. lower")
    color = st.text_input("Color (optional — enables color QA)")
with c2:
    back_file = st.file_uploader("Back image (optional)", type=["jpg", "jpeg", "png", "webp"])
    composition = st.text_input("Material composition (optional)", placeholder="e.g. 95% Cotton 5% Elastane")
    brand_color = st.text_input("Brand color code (optional)", placeholder="e.g. LGY")
    season = st.text_input("Season (optional)")

c3, c4 = st.columns(2)
sku_id = c3.text_input("SKU ID (optional)")
ean_code = c4.text_input("EAN code (optional)")

if front_file:
    st.image(front_file, caption="Front", width=200)
if back_file:
    st.image(back_file, caption="Back", width=200)

st.header("2. Run")
run_a, run_b, run_both = st.columns(3)
do_a = run_a.button("▶ Run Model A", use_container_width=True)
do_b = run_b.button("▶ Run Model B", use_container_width=True)
do_both = run_both.button("▶ Run Both & Compare", use_container_width=True, type="primary")


def gather_inputs() -> Optional[Dict[str, Any]]:
    if not front_file:
        st.error("A front image is required.")
        return None
    return {
        "front_uri": file_to_data_uri(front_file),
        "back_uri": file_to_data_uri(back_file),
        "category": category,
        "subcategory": subcategory,
        "color": color,
        "composition": composition,
        "brand_color": brand_color,
        "season": season,
        "sku_id": sku_id,
        "ean_code": ean_code,
    }


def render_result(label: str, output: Dict[str, Any]):
    st.subheader(label)
    result = output["result"]
    summary = output["token_summary"]

    st.metric("Estimated cost (this run)", f"${summary['estimated_cost_usd']:.6f}")
    m1, m2 = st.columns(2)
    m1.metric("Input tokens", summary["total_input_tokens"])
    m2.metric("Output tokens", summary["total_output_tokens"])

    st.markdown(f"**Product name:** {result['product_name']}")
    if result.get("ai_color") is not None or result.get("color_flag") is not None:
        st.markdown(f"**Color QA:** claimed=`{result['prominent_color']}` · ai_observed=`{result.get('ai_color')}` · flag=`{result.get('color_flag')}`")

    with st.expander("Extracted attributes", expanded=False):
        st.json(result["extracted_attributes"])

    with st.expander("Product description + keywords", expanded=False):
        st.write(result["product_description"].get("paragraph", ""))
        st.write("**Generic keywords:**", ", ".join(result["product_description"].get("generic_keywords", [])))
        st.write("**Short keywords:**", ", ".join(result["product_description"].get("short_keywords", [])))

    with st.expander("Per-step token breakdown", expanded=False):
        st.table(summary["steps"])

    with st.expander("Raw result dict", expanded=False):
        st.json(result)


def run_and_render(cfg, label):
    inputs = gather_inputs()
    if inputs is None:
        return
    try:
        with st.spinner(f"Running {label}…"):
            output = run_pipeline(cfg, inputs)
        render_result(label, output)
        return output
    except Exception as e:
        st.error(f"{label} failed: {e}")
        return None


st.header("3. Results")

if do_a:
    run_and_render(cfg_a, "Model A")

elif do_b:
    run_and_render(cfg_b, "Model B")

elif do_both:
    col_a, col_b = st.columns(2)
    with col_a:
        out_a = run_and_render(cfg_a, "Model A")
    with col_b:
        out_b = run_and_render(cfg_b, "Model B")

    if out_a and out_b:
        st.header("4. Cost comparison")
        cost_a = out_a["token_summary"]["estimated_cost_usd"]
        cost_b = out_b["token_summary"]["estimated_cost_usd"]
        diff = cost_a - cost_b
        st.write(
            f"Model A cost **${cost_a:.6f}** vs Model B cost **${cost_b:.6f}** "
            f"— difference: **${abs(diff):.6f}** ({'A cheaper' if diff < 0 else 'B cheaper' if diff > 0 else 'equal'})."
        )
