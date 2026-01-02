import requests
from bs4 import BeautifulSoup
import os

# List of URLs for questions 21 to 30
urls = [
    # Replace these URLs with the actual question URLs
    'https://www.examtopics.com/discussions/snowflake/view/70974-exam-snowpro-core-topic-1-question-21-discussion/',
    'https://www.examtopics.com/discussions/snowflake/view/70975-exam-snowpro-core-topic-1-question-22-discussion/',
    'https://www.examtopics.com/discussions/snowflake/view/70976-exam-snowpro-core-topic-1-question-23-discussion/',
    'https://www.examtopics.com/discussions/snowflake/view/70977-exam-snowpro-core-topic-1-question-24-discussion/',
    'https://www.examtopics.com/discussions/snowflake/view/70978-exam-snowpro-core-topic-1-question-25-discussion/',
    'https://www.examtopics.com/discussions/snowflake/view/70979-exam-snowpro-core-topic-1-question-26-discussion/',
    'https://www.examtopics.com/discussions/snowflake/view/70980-exam-snowpro-core-topic-1-question-27-discussion/',
    'https://www.examtopics.com/discussions/snowflake/view/70981-exam-snowpro-core-topic-1-question-28-discussion/',
    'https://www.examtopics.com/discussions/snowflake/view/70982-exam-snowpro-core-topic-1-question-29-discussion/',
    'https://www.examtopics.com/discussions/snowflake/view/70983-exam-snowpro-core-topic-1-question-30-discussion/',
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

output_dir = "examtopics_questions"
os.makedirs(output_dir, exist_ok=True)

for url in urls:
    print(f"Processing: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch {url} - Status code: {response.status_code}")
        continue

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract title
    title_tag = soup.find('h1')
    title = title_tag.text.strip() if title_tag else "No title found"

    # Extract question text
    question_div = soup.find('div', class_='question-body mt-3 pt-3 border-top')
    question_text = question_div.get_text(separator='\n').strip() if question_div else "No question content found."

    # Extract choices
    choices_container = soup.find('div', class_='question-choices-container')
    choices = []
    if choices_container:
        choice_divs = choices_container.find_all('div', class_='card-text question-answer bg-light white-text')
        for choice_div in choice_divs:
            choices.append(choice_div.get_text(strip=True))

    # Extract answers/comments
    comments_section = soup.find('div', class_='discussion-page-comments-section')
    answers = []
    if comments_section:
        comment_bodies = comments_section.find_all('div', class_='comment-body comment-toggled')
        for comment in comment_bodies:
            content_div = comment.find('div', class_='comment-content')
            if content_div:
                answers.append(content_div.text.strip())

    # Prepare filename based on question number from URL (assumes question number in URL)
    import re
    match = re.search(r'question-(\d+)-discussion', url)
    q_num = match.group(1) if match else 'unknown'

    filename = os.path.join(output_dir, f"question_{q_num}.txt")

    with open(filename, 'w', encoding='utf-8') as f:
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

    print(f"Saved question {q_num} to {filename}\n")

print("Done extracting all questions.")
