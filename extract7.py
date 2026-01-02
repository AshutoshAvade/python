from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import traceback

# Output file (single file for all questions)
output_file = "examtopics_all_questions.txt"

# If file exists from earlier run, clear it first
open(output_file, "w", encoding="utf-8").close()

# Selenium setup
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)
# options.add_argument("--headless=new")  # Uncomment for headless mode

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

def search_google_and_open_examtopics(question_number):
    """Search Google and navigate to the first ExamTopics link for the question."""
    driver.get("https://www.google.com")
    print(f"Searching Google for: examtopics snowpro core question {question_number}")

    try:
        # Wait for search input
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        search_box.clear()
        search_box.send_keys(f"examtopics snowpro core question {question_number}")
        search_box.submit()

        # Wait for first ExamTopics link
        exam_link_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(@href, 'examtopics.com/discussions')]"))
        )
        url = exam_link_element.get_attribute("href")
        print(f"Found ExamTopics link: {url}")
        driver.get(url)
        return url
    except Exception as e:
        print(f"Failed to find ExamTopics link for question {question_number}: {e}")
        traceback.print_exc()
        return None

def extract_question_content():
    """Extract title, question, choices, comments from ExamTopics page."""
    # Title
    try:
        title_tag = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        title = title_tag.text.strip() if title_tag else "No title found"
    except TimeoutException:
        title = "No title found"

    # Question text
    try:
        question_div = driver.find_element(By.CSS_SELECTOR, "div.question-body, div[class*='question-text']")
        question_text = question_div.text.strip()
    except NoSuchElementException:
        question_text = "No question content found."

    # Choices/options
    choices = []
    try:
        choices_container = driver.find_element(By.CSS_SELECTOR, "div.question-choices-container, ul.answers-list")
        choice_divs = choices_container.find_elements(By.CSS_SELECTOR, "div.card-text, li")
        for c in choice_divs:
            text = c.text.strip()
            if text:
                choices.append(text)
    except NoSuchElementException:
        pass

    # Comments / answers
    answers = []
    try:
        comments_section = driver.find_element(By.CSS_SELECTOR, "div.discussion-page-comments-section")
        comment_bodies = comments_section.find_elements(By.CSS_SELECTOR, "div.comment-body")
        for comment in comment_bodies:
            try:
                content_div = comment.find_element(By.CSS_SELECTOR, "div.comment-content")
                answers.append(content_div.text.strip())
            except NoSuchElementException:
                continue
    except NoSuchElementException:
        pass

    return title, question_text, choices, answers

# Main loop
start_q =788 
end_q = 847

for qnum in range(start_q, end_q + 1):
    print(f"\n--- Processing question {qnum} ---")
    try:
        url = search_google_and_open_examtopics(qnum)
        if not url:
            continue

        time.sleep(2)  # Allow page to fully load

        title, question_text, choices, answers = extract_question_content()

        # Append results to one single file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Question {qnum}\n")
            f.write(f"Title: {title}\n\n")
            f.write(f"Question:\n{question_text}\n\n")
            if choices:
                f.write("Choices:\n")
                for idx, c in enumerate(choices, 1):
                    f.write(f"{idx}. {c}\n")
                f.write("\n")
            if answers:
                f.write(f"Answers/Comments ({len(answers)} found):\n")
                for idx, ans in enumerate(answers, 1):
                    f.write(f"{idx}. {ans}\n\n")

            f.write("--------------------------------------------------------------------------------------------------------\n\n")

        print(f"Appended question {qnum} to {output_file}")

    except Exception as e:
        print(f"Error processing question {qnum}: {e}")
        traceback.print_exc()

driver.quit()
print(f"\nDone extracting all questions. Saved to {output_file}")
