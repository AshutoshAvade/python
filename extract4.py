from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def extract_question_content(driver):
    try:
        # Wait up to 10 seconds for main question container (update selector here)
        question_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.discussion-content, div.content, div.question, div.post-content"))
        )
        text = question_container.text.strip()
        if text:
            return text
    except Exception as e:
        print(f"Primary selector failed: {e}")

    # Fallback: grab all visible text in body (avoid ads or nav)
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        text = body.text.strip()
        return text
    except Exception as e:
        print(f"Fallback extraction failed: {e}")
        return ""

def main():
    options = Options()
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    for qnum in range(21, 31):
        try:
            print(f"\n--- Processing question {qnum} ---")
            # Search google with site and question number
            search_query = f"examtopics snowpro core question {qnum}"
            driver.get(f"https://www.google.com/search?q={search_query}")

            # Wait for search results links to appear
            results = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.yuRUbf > a"))
            )

            # Find first examtopics.com link
            exam_link = None
            for res in results:
                href = res.get_attribute("href")
                if "examtopics.com/discussions" in href:
                    exam_link = href
                    break

            if not exam_link:
                print(f"No examtopics link found for question {qnum}")
                continue

            print(f"Opening link for question {qnum}: {exam_link}")
            driver.get(exam_link)

            # Wait a little to allow content (avoid ads if possible)
            time.sleep(2)

            # Extract question content
            content = extract_question_content(driver)

            if content:
                print(f"\nQuestion {qnum} content:\n{content[:1000]}")  # Print first 1000 chars to avoid overload
            else:
                print(f"Question content not found for question {qnum}.")

        except Exception as e:
            print(f"Error processing question {qnum}: {e}")

    driver.quit()

if __name__ == "__main__":
    main()
