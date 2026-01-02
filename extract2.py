import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

EXAM_QUESTIONS_URL = "https://www.examtopics.com/exams/snowflake/snowpro-core/view/"

def get_question_page_urls(start=21, end=30):
    print(f"Starting browser to load question list from {start} to {end}...")

    options = Options()
    options.headless = False  # Set True to hide browser UI
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(EXAM_QUESTIONS_URL)
        print("Page loaded, waiting for questions container...")

        # Wait for the container that holds questions - update this selector if needed
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.exam-question-list"))
        )
        print("Questions container found.")

        time.sleep(2)  # Let page fully load

        # Find all question elements inside container
        question_elements = driver.find_elements(By.CSS_SELECTOR, "div.exam-question-list > div")

        print(f"Found {len(question_elements)} question entries on page.")

        question_urls = {}

        for elem in question_elements:
            # Each question might have a number and link somewhere inside
            try:
                q_num_text = elem.find_element(By.CSS_SELECTOR, "span.question-number").text.strip()
                q_num = int(q_num_text.replace("#", ""))  # e.g. "#21" -> 21
            except Exception:
                # Skip elements without question number or parse error
                continue

            if start <= q_num <= end:
                try:
                    link_elem = elem.find_element(By.CSS_SELECTOR, "a.question-link")
                    href = link_elem.get_attribute("href")
                except Exception:
                    href = None

                if href:
                    question_urls[q_num] = href
                    print(f"Question {q_num}: {href}")

        print(f"Extracted {len(question_urls)} questions between {start} and {end}.")
        return question_urls

    finally:
        driver.quit()

def fetch_question_text(url):
    options = Options()
    options.headless = True
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.question-content"))
        )
        time.sleep(1)  # Wait for content

        question_div = driver.find_element(By.CSS_SELECTOR, "div.question-content")
        question_text = question_div.text.strip()

        # Extract options if available
        options_div = driver.find_elements(By.CSS_SELECTOR, "div.question-answers div.answer")
        options_texts = [opt.text.strip() for opt in options_div]

        return question_text, options_texts

    finally:
        driver.quit()

def main():
    start_q = 21
    end_q = 30

    question_urls = get_question_page_urls(start_q, end_q)
    if not question_urls:
        print("No questions found in the specified range.")
        return

    for q_num in range(start_q, end_q + 1):
        url = question_urls.get(q_num)
        if not url:
            print(f"Question {q_num} URL not found.")
            continue

        print(f"\nFetching Question {q_num} from {url}")
        question_text, options = fetch_question_text(url)
        print(f"Question {q_num} Text:\n{question_text}\n")
        if options:
            print("Options:")
            for idx, opt in enumerate(options, start=1):
                print(f"  {idx}. {opt}")

if __name__ == "__main__":
    main()
