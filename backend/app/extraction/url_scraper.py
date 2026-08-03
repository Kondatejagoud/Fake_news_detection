import trafilatura
from newspaper import Article
from app.core.logging import logger

def scrape_url(url: str) -> dict:
    """
    Extracts article text, page title, images, and metadata from a URL.
    Uses trafilatura as primary text extractor and newspaper3k as fallback/metadata retriever.
    """
    logger.info(f"URL Scraper: scraping {url}")
    result = {
        "text": "",
        "title": "",
        "images": [],
        "metadata": {}
    }

    # 1. Newspaper3k for parsing metadata, title, and images
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        result["title"] = article.title
        result["metadata"] = {
            "authors": article.authors,
            "publish_date": str(article.publish_date) if article.publish_date else None,
            "summary": article.summary or "",
            "meta_description": article.meta_description or "",
            "keywords": article.keywords or []
        }
        
        # Collect top image and all images
        images = []
        if article.top_image:
            images.append(article.top_image)
        for img in article.images:
            if img not in images:
                images.append(img)
        result["images"] = images
        result["text"] = article.text  # Default fallback text
        logger.info(f"Newspaper3k successfully parsed article. Title: '{article.title}'")
    except Exception as e:
        logger.warning(f"Newspaper3k failed to parse URL {url}: {e}")

    # 2. Trafilatura for premium clean text extraction
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted_text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            if extracted_text and len(extracted_text) > len(result["text"]):
                result["text"] = extracted_text
                logger.info("Trafilatura successfully extracted clean text.")
                
            # Grab additional metadata if newspaper failed
            if not result["title"]:
                metadata = trafilatura.extract_metadata(downloaded)
                if metadata:
                    result["title"] = metadata.title or ""
                    result["metadata"].update({
                        "publish_date": metadata.date or result["metadata"].get("publish_date"),
                        "sitename": metadata.sitename or ""
                    })
    except Exception as e:
        logger.warning(f"Trafilatura failed to extract text from URL {url}: {e}")

    # 3. Clean up and finalize result
    if not result["text"].strip():
        # If both scrapers failed, throw/return warning
        result["text"] = "Warning: Scraper could not extract body text. The site may be protected by anti-scraping layers."
        
    logger.info(f"Scraped content length: {len(result['text'])} characters. Extracted {len(result['images'])} images.")
    return result
