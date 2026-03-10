import os
import time
import pickle

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


COOKIES_FILE = "amazon_cookies.pkl"


# -------------------------------
#  Launch Selenium Browser
# -------------------------------
def launch_browser():
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0")
    # options.add_argument("--headless")  # optional

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    driver.implicitly_wait(5)
    return driver


# -------------------------------
#  Manual Login + Save Cookies
# -------------------------------
def login_and_save_cookies():
    driver = launch_browser()

    print("🔐 Opening Amazon login page — please login manually.")
    driver.get("https://www.amazon.ae/ap/signin")

    input("👉 After fully logged in, press ENTER to save cookies...")

    cookies = driver.get_cookies()
    pickle.dump(cookies, open(COOKIES_FILE, "wb"))
    print("✔ Cookies saved to amazon_cookies.pkl")

    driver.quit()


# -------------------------------
#  Load Cookies Into Driver
# -------------------------------
def load_cookies(driver):
    cookies = pickle.load(open(COOKIES_FILE, "rb"))

    driver.get("https://www.amazon.ae/")
    time.sleep(2)

    for cookie in cookies:
        cookie.pop("sameSite", None)  # fix selenium edge-case
        try:
            driver.add_cookie(cookie)
        except:
            pass

    driver.get("https://www.amazon.ae/")
    time.sleep(2)
    print("✔ Cookies loaded — session restored.")


# -------------------------------
#  Scrape ALL Review Pages
# -------------------------------
def get_all_reviews(asin, max_pages=50):
    if not os.path.exists(COOKIES_FILE):
        print("❌ No cookies found — starting login flow.")
        login_and_save_cookies()

    driver = launch_browser()
    load_cookies(driver)

    url = f"https://www.amazon.ae/product-reviews/{asin}/?sortBy=recent&reviewerType=all_reviews"
    driver.get(url)
    time.sleep(2)

    all_reviews = []
    page = 1

    while page <= max_pages:
        print(f"\n📄 Page {page}")

        # Extract reviews
        blocks = driver.find_elements(By.CSS_SELECTOR, "li[data-hook='review']")
        print("   → Reviews on page:", len(blocks))

        if len(blocks) == 0:
            print("❌ No more reviews. Stopping.")
            break

        for block in blocks:
            try:
                txt = block.find_element(By.CSS_SELECTOR, "[data-hook='review-body']").text.strip()
                all_reviews.append(txt)
            except:
                pass

        # Try clicking next page
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "li.a-last a")
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(2)
            page += 1
        except:
            print("✔ Reached last page.")
            break

    driver.quit()
    return all_reviews


# -------------------------------
#  MAIN
# -------------------------------
if __name__ == "__main__":
    asin = "B0DPQW3VH6"  # your test ASIN

    reviews = get_all_reviews(asin)

    print("\n============================")
    print("TOTAL REVIEWS SCRAPED:", len(reviews))
    print("============================\n")

    # Show sample
    for r in reviews[:10]:
        print("----")
        print(r)
