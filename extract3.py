from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time

def search_google_and_open_examtopics(driver, question_number):
    driver.get("https://www.google.com")
    print(f"Searching Google for: examtopics snowpro core question {question_number}")

    # Wait for search input box and type query
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "q"))
    )
    search_box.clear()
    search_box.send_keys(f"examtopics snowpro core question {question_number}")
    search_box.submit()

    # Wait for results to load and find first examtopics link
    examtopics_link = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//a[contains(@href, 'examtopics.com')]")
        )
    )

    # Open first examtopics link in same tab
    url = examtopics_link.get_attribute("href")
    print(f"Opening link for question {question_number}: {url}")
    driver.get(url)

def extract_question_content(driver):
    try:
        # Wait max 3 seconds for question container (adjust selector as needed)
        question_container = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[class*='question-content'], div[class*='question-text']")
            )
        )
        question_text = question_container.text.strip()
    except TimeoutException:
        print("Question content not found within 3 seconds.")
        question_text = ""

    options = []
    try:
        # Try to find answers/options container
        options_container = driver.find_element(
            By.CSS_SELECTOR, "ul.answers-list, div.answers, div[class*='answers']"
        )
        # Get all <li> inside options container
        option_elements = options_container.find_elements(By.TAG_NAME, "li")
        if not option_elements:
            # If no <li>, try labels or divs with class option
            option_elements = options_container.find_elements(By.CSS_SELECTOR, "label, div.option")
        for opt in option_elements:
            text = opt.text.strip()
            if text:
                options.append(text)
    except (NoSuchElementException, TimeoutException):
        print("No options container or options found.")

    return question_text, options

def main():
    # Setup driver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.maximize_window()

    start_q = 21
    end_q = 30

    for qnum in range(start_q, end_q + 1):
        print(f"\n--- Processing question {qnum} ---")
        try:
            search_google_and_open_examtopics(driver, qnum)

            # Give a tiny delay to let page settle (adjust if needed)
            time.sleep(1)

            question_text, options = extract_question_content(driver)

            print(f"\nQuestion {qnum}:\n{question_text}\nOptions:")
            if options:
                for idx, opt in enumerate(options, 1):
                    print(f"  {idx}. {opt}")
            else:
                print("  No options found.")

        except Exception as e:
            print(f"Error processing question {qnum}: {e}")

    driver.quit()

if __name__ == "__main__":
    main()
