import difflib
import io
import re
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from rapidocr import RapidOCR

ocr_engine = RapidOCR()
CAPITAL_TO_COUNTRIES: dict[str, list[str]] = {}
http_client: httpx.AsyncClient | None = None

COUNTRY_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "uk": "united kingdom",
    "uae": "united arab emirates",
    "russia": "russian federation",
    "czechia": "czech republic",
    "ivory coast": "cote divoire",
    "drc": "democratic republic of the congo",
    "dr congo": "democratic republic of the congo",
    "vatican": "vatican city",
    "holland": "netherlands",
    "burma": "myanmar",
    "turkiye": "turkey",
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def ocr_fuzzy_norm(text: str) -> str:
    """Treat 'l' and 'i' as equivalent to handle RapidOCR font confusion."""
    return normalize(text).replace("l", "i")


def add_capital_mapping(capital: str, country: str):
    key = normalize(capital)
    if not key or not country:
        return
    bucket = CAPITAL_TO_COUNTRIES.setdefault(key, [])
    if country not in bucket:
        bucket.append(country)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=15, follow_redirects=True)

    # 1. Primary source: RestCountries (clean English capital & country names)
    try:
        resp = await http_client.get("https://restcountries.com/v3.1/all?fields=name,capital")
        resp.raise_for_status()
        for entry in resp.json():
            name_obj = entry.get("name") or {}
            common = name_obj.get("common") or ""
            official = name_obj.get("official") or ""
            for cap in entry.get("capital") or []:
                if common:
                    add_capital_mapping(cap, common)
                if official and official != common:
                    add_capital_mapping(cap, official)
    except Exception as e:
        print(f"[Warn] RestCountries load failed: {e}")

    # 2. Secondary source: samayo/country-json (adds native/alternate spellings)
    try:
        resp = await http_client.get(
            "https://raw.githubusercontent.com/samayo/country-json/master/src/country-by-capital-city.json"
        )
        resp.raise_for_status()
        for entry in resp.json():
            country = entry.get("country") or ""
            cap = entry.get("city") or ""
            # Handle bracketed names like "Bruxelles [Brussel]"
            for part in re.split(r"[\[\]/]", cap):
                add_capital_mapping(part, country)
    except Exception as e:
        print(f"[Warn] samayo dataset load failed: {e}")

    # Manual extra aliases for common quiz variants
    extra = {
        "washington dc": "United States",
        "washington": "United States",
        "muscat": "Oman",
        "brussels": "Belgium",
        "copenhagen": "Denmark",
        "vienna": "Austria",
        "kyiv": "Ukraine",
        "kiev": "Ukraine",
        "astana": "Kazakhstan",
    }
    for cap, country in extra.items():
        add_capital_mapping(cap, country)

    ocr_engine(np.zeros((64, 64, 3), dtype=np.uint8), use_cls=False)

    print(f"Loaded {len(CAPITAL_TO_COUNTRIES)} capital keys. OCR warmed up and ready.")
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    image_url: str


def extract_ocr_items(img_array: np.ndarray) -> list[dict]:
    res = ocr_engine(img_array, use_cls=False)
    boxes = getattr(res, "boxes", None)
    txts = getattr(res, "txts", None)
    if boxes is None or txts is None:
        return []

    items = []
    for box, txt in zip(boxes, txts):
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        items.append({
            "text": txt.strip(),
            "x": sum(xs) / len(xs),
            "y": sum(ys) / len(ys),
            "x_min": min(xs),
            "x_max": max(xs),
        })
    return items


def parse_card(items: list[dict], width: int, height: int) -> tuple[str | None, dict[str, str]]:
    capital = None
    quadrants: dict[str, list[dict]] = {"A": [], "B": [], "C": [], "D": []}

    for item in items:
        m = re.search(r"capital\s*[:\-]\s*(.+)", item["text"], re.IGNORECASE)
        if m:
            capital = m.group(1).strip()
            break

    for item in items:
        if item["y"] < height * 0.42:
            continue

        txt = item["text"].strip()
        txt = re.sub(r"^[ABCD]\s+", "", txt)
        if not txt or txt in {"A", "B", "C", "D"}:
            continue

        is_top = item["y"] < height * 0.70
        is_left = item["x"] < width * 0.50

        if is_top and is_left:
            slot = "A"
        elif is_top and not is_left:
            slot = "B"
        elif not is_top and is_left:
            slot = "C"
        else:
            slot = "D"

        quadrants[slot].append({"text": txt, "x_min": item["x_min"]})

    options: dict[str, str] = {}
    for slot, parts in quadrants.items():
        if parts:
            parts.sort(key=lambda p: p["x_min"])
            options[slot] = " ".join(p["text"] for p in parts)

    return capital, options


def pick_best_option(capital: str, options: dict[str, str]) -> tuple[str | None, str | None]:
    norm_cap = normalize(capital)

    if norm_cap in CAPITAL_TO_COUNTRIES:
        cap_key = norm_cap
    else:
        cap_matches = difflib.get_close_matches(norm_cap, CAPITAL_TO_COUNTRIES.keys(), n=1, cutoff=0.72)
        if cap_matches:
            cap_key = cap_matches[0]
        else:
            fuzzy_cap = ocr_fuzzy_norm(norm_cap)
            fuzzy_map = {ocr_fuzzy_norm(k): k for k in CAPITAL_TO_COUNTRIES.keys()}
            fuzzy_matches = difflib.get_close_matches(fuzzy_cap, fuzzy_map.keys(), n=1, cutoff=0.75)
            if not fuzzy_matches:
                return None, None
            cap_key = fuzzy_map[fuzzy_matches[0]]

    target_countries = [normalize(c) for c in CAPITAL_TO_COUNTRIES[cap_key]]

    best_letter = None
    best_score = 0.0
    best_country = None

    for letter, opt_text in options.items():
        norm_opt = normalize(opt_text)
        norm_opt = COUNTRY_ALIASES.get(norm_opt, norm_opt)
        fuzzy_opt = ocr_fuzzy_norm(norm_opt)

        for target in target_countries:
            fuzzy_target = ocr_fuzzy_norm(target)
            if len(fuzzy_opt) >= 4 and (fuzzy_opt in fuzzy_target or fuzzy_target in fuzzy_opt):
                score = 0.95
            else:
                score = max(
                    difflib.SequenceMatcher(None, norm_opt, target).ratio(),
                    difflib.SequenceMatcher(None, fuzzy_opt, fuzzy_target).ratio(),
                )

            if score > best_score:
                best_score = score
                best_letter = letter
                best_country = opt_text

    if best_score >= 0.6:
        return best_letter, best_country

    missing = [l for l in ("A", "B", "C", "D") if l not in options]
    if len(missing) == 1:
        return missing[0], CAPITAL_TO_COUNTRIES[cap_key][0]

    return None, None


@app.post("/solve")
async def solve(req: SolveRequest):
    t0 = time.perf_counter()
    resp = await http_client.get(req.image_url)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image")
    t1 = time.perf_counter()

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    w, h = img.size

    img = img.crop((0, int(h * 0.30), w, int(h * 0.86)))

    if img.width > 768:
        ratio = 768 / img.width
        img = img.resize((768, int(img.height * ratio)), Image.Resampling.BILINEAR)

    items = extract_ocr_items(np.array(img))
    capital, options = parse_card(items, img.width, img.height)
    t2 = time.perf_counter()

    letter, country = pick_best_option(capital, options) if (capital and options) else (None, None)
    print(f"[Solve] download={(t1 - t0)*1000:.0f}ms | ocr={(t2 - t1)*1000:.0f}ms | {capital} -> {country} ({letter}) | opts={options}")

    return {
        "answer": letter,
        "country": country,
        "capital": capital,
        "options": options,
    }