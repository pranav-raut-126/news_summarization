from newspaper import Article


def fetch_article_data(url: str) -> dict:
    """
    Fetches and parses a news article from a given URL.

    Returns a dictionary containing:
    - title
    - full text
    - top image URL
    - authors
    - publish date
    - success flag
    """

    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()   # improves extraction quality

        # Validate extracted text
        if not article.text or len(article.text.strip()) < 200:
            return {
                "success": False,
                "error": "Extracted article text is too short or empty."
            }

        return {
            "success": True,
            "title": article.title,
            "text": article.text,
            "top_image": article.top_image,
            "authors": article.authors,
            "publish_date": article.publish_date
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# --- Quick standalone test ---
if __name__ == "__main__":
    test_url = input("Paste the URL here : ")
    result = fetch_article_data(test_url)

    if result["success"]:
        print("✅ Article fetched successfully")
        print("Title:", result["title"])
        print("Text preview:", result["text"])
    else:
        print("❌ Failed to fetch article")
        print("Error:", result["error"])
