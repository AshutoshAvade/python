from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import time
import traceback

def get_examtopics_link(question_num):
    """Search Google for the ExamTopics discussion link for a given question number."""
    search_query = f"examtopics snowpro core question {question_num}"
    url = f"https://www.google.com/search?q={search_query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/140.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Google search results links
        for a_tag in soup.select("a"):
            href = a_tag.get("href")
            if href and "examtopics.com/discussions" in href:
                # Clean up link
                if href.startswith("/url?q="):
                    href = href.split("/url?q=")[1].split("&")[0]
                return href
    except Exception as e:
        print(f"Error fetching Google search for question {question_num}: {e}")
        traceback.print_exc()
    return None

def extract_question_content(driver):
    """Extract main question content from the ExamTopics page."""
    try:
        question_container = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.discussion-content, div.content, div.question, div.post-content")
            )
        )
        text = question_container.text.strip()
        if text:
            return text
    except Exception as e:
        print(f"Primary selector failed: {e}")
        traceback.print_exc()
    
    # Fallback: grab all visible text in body
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        return body.text.strip()
    except Exception as e:
        print(f"Fallback extraction failed: {e}")
        traceback.print_exc()
        return ""

def main():
    # Selenium options
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    )
    # options.add_argument("--headless=new")  # uncomment for headless

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    for qnum in range(21, 31):
        try:
            print(f"\n--- Processing question {qnum} ---")
            exam_link = get_examtopics_link(qnum)
            
            if not exam_link:
                print(f"No ExamTopics link found for question {qnum}")
                continue
            
            print(f"Opening ExamTopics link for question {qnum}: {exam_link}")
            driver.get(exam_link)

            # Wait a few seconds for content to load
            time.sleep(2)

            content = extract_question_content(driver)
            if content:
                print(f"\nQuestion {qnum} content (first 1000 chars):\n{content[:1000]}")
            else:
                print(f"Question content not found for question {qnum}")

        except Exception as e:
            print(f"Error processing question {qnum}: {e}")
            traceback.print_exc()

    driver.quit()

if __name__ == "__main__":
    main()
