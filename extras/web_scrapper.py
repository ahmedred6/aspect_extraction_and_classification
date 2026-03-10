import json
import re
import time
import os
import pickle
import nltk
from nltk.corpus import words
from difflib import get_close_matches

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# =============== 1. DOWNLOAD ENGLISH CORPUS ===================
nltk.download("words")
ENGLISH_CORPUS = set(words.words())

# =============== 2. ELECTRONICS KEYWORDS ======================
ELECTRONICS_KEYWORDS = [
    "Laptop"
]

COOKIES_FILE = "amazon_cookies.pkl"
OUTPUT_FILE = "amazon_reviews_bilingual.jsonl"

# =============== 3. LANGUAGE DETECTORS ========================

EURO_ACCENTS = r"[àâäéèêëîïôöùûüÿçñ]"
BLACKLIST_SHORT = {"il","en","la","le","de","des","du","et","un","une","je",
                   "tu","te","se","sur","par","pour","mais","si","lo","el",
                   "con","que","mi","es","yo"}

def tokenize(text):
    return re.findall(r"[A-Za-z]+", text.lower())

def is_arabic(text):
    return re.search(r"[\u0600-\u06FF]", text) is not None

def is_english(text, threshold=0.60):
    # Rule 1: Arabic overrides
    if is_arabic(text):
        return False

    # Rule 2: Accents → not English
    if re.search(EURO_ACCENTS, text):
        return False

    tokens = tokenize(text)
    if not tokens:
        return False

    filtered = [t for t in tokens if t not in BLACKLIST_SHORT]
    if not filtered:
        return False

    # Exact corpus matches
    exact = sum(1 for t in filtered if t in ENGLISH_CORPUS)

    # Fuzzy correction for misspellings
    fuzzy = 0
    for t in filtered:
        if get_close_matches(t, ENGLISH_CORPUS, n=1, cutoff=0.8):
            fuzzy += 1

    score = (exact + fuzzy) / len(filtered)
    return score >= threshold

# =============== 4. BROWSER SETUP =============================

def launch_browser():
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.implicitly_wait(5)
    return driver

def login_and_save_cookies():
    driver = launch_browser()
    driver.get("https://www.amazon.ae/ap/signin")
    input("Login manually, then press ENTER...")
    pickle.dump(driver.get_cookies(), open(COOKIES_FILE, "wb"))
    driver.quit()

def load_cookies(driver):
    cookies = pickle.load(open(COOKIES_FILE, "rb"))
    driver.get("https://www.amazon.ae/")
    time.sleep(2)
    for c in cookies:
        c.pop("sameSite", None)
        try:
            driver.add_cookie(c)
        except:
            pass
    driver.get("https://www.amazon.ae/")
    time.sleep(2)

# =============== 5. SCRAPING HELPERS ==========================

def get_asins(keyword, max_items=10):
    if not os.path.exists(COOKIES_FILE):
        login_and_save_cookies()

    driver = launch_browser()
    load_cookies(driver)

    search_url = f"https://www.amazon.ae/s?k={keyword.replace(' ', '+')}"
    driver.get(search_url)
    time.sleep(2)

    blocks = driver.find_elements(By.CSS_SELECTOR, "div[data-component-type='s-search-result']")
    asins = []

    for b in blocks:
        asin = b.get_attribute("data-asin")
        if asin and len(asin) == 10:
            asins.append(asin)
        if len(asins) >= max_items:
            break

    driver.quit()
    return asins

def append_review(obj):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# =============== 6. SCRAPE REVIEWS (Arabic + English) =========

def scrape_reviews(asin, max_pages=50):
    driver = launch_browser()
    load_cookies(driver)

    url = f"https://www.amazon.ae/product-reviews/{asin}/?sortBy=recent&reviewerType=all_reviews"
    driver.get(url)

    page = 1
    while page <= max_pages:
        blocks = driver.find_elements(By.CSS_SELECTOR, "li[data-hook='review']")

        if not blocks:
            break

        for b in blocks:
            try:
                text = b.find_element(By.CSS_SELECTOR, "[data-hook='review-body']").text.strip()

                if is_arabic(text) or is_english(text):
                    lang = "arabic" if is_arabic(text) else "english"
                    append_review({
                        "asin": asin,
                        "page": page,
                        "lang": lang,
                        "text": text
                    })
                    print(f"[{lang}] Saved review")

            except:
                pass

        # Next page
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "li.a-last a")
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(2)
            page += 1
        except:
            break

    driver.quit()

# =============== 7. MAIN PIPELINE =============================

def run_pipeline():
    open(OUTPUT_FILE, "w", encoding="utf-8").close()
    print("▶ Starting bilingual electronics scraper...")

    for keyword in ELECTRONICS_KEYWORDS:
        print(f"\n🔍 Searching products for keyword: {keyword}")
        asins = get_asins(keyword)

        for asin in asins:
            print(f"\n📦 Scraping reviews for ASIN {asin}")
            scrape_reviews(asin)

    print("\n🎉 DONE — Saved bilingual reviews to:", OUTPUT_FILE)

run_pipeline()
