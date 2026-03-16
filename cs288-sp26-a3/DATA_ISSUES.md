# Data Investigation Report: EECS Website Raw Data

This document identifies issues found in the raw scraped data from `eecs.berkeley.edu` that need to be addressed during the **Cleaning** phase.

## 1. High Redundancy & Duplicate Files
The current crawler logic generates multiple files for the same URL due to filename collisions or re-visiting.
- **Example**: `category_people_faculty.json`, `category_people_faculty_1.json`, and `category_people_faculty_2.json` are identical.
- **Problem**: Increases indexing time and retrieval noise.
- **Fix**: Deduplicate based on the `url` field before processing.

## 2. Boilerplate & Navigation Noise
Many pages contain repetitive uninformative strings at the end or within the content.
- **Patterns identified**:
    - `"1 2 Next Page \u00bb"`
    - `"View Open Faculty Positions"`
    - `"Faculty Archives - EECS at Berkeley Faculty"` (Boilerplate titles)
    - Social media placeholders or breadcrumbs.
- **Problem**: These strings do not provide answers to QA and can pollute the embedding space.
- **Fix**: Use regex-based cleaning to strip known boilerplate suffixes and headers.

## 3. Archive/Category Pages vs. Source Pages
Pages like `https://eecs.berkeley.edu/category/people/faculty/` are archive listings.
- **Problem**: They contain truncated summaries of news/profiles. The "real" information is on the linked full-article page.
- **Risk**: A retriever might find the truncated version instead of the full version.
- **Fix**: Prioritize individual profile/news pages; consider lowering the weight or excluding "Archive" pages if the full content is already captured elsewhere.

## 4. Multi-Page Lists (Pagination)
Some directories (like faculty lists) span multiple actual URLs (e.g., `.../page/2/`).
- **Investigation**: We need to verify if the crawler is correctly identifying these as unique content or if it's getting stuck on the first page.
- **Fix**: Ensure the crawler follows `Next Page` links and that the processor treats these as a continuation of a single "Datastore" entity if appropriate.

## 5. Metadata Noise
The content contains escaped characters and date strings (e.g., `\u201c`, `\u201d`, `February 17, 2026`).
- **Fix**: Normalize unicode characters and decide if dates are relevant for the QA task (some questions are temporal, e.g., "winner in 2024").

## 6. Truncated Content
Many archive entries end with `\u2026` (ellipsis).
- **Problem**: Information is cut off.
- **Fix**: Ensure we have the full pages for these summaries.

---
**Next Step**: Implement a `cleaner.py` that handles deduplication (by URL) and boilerplate removal using the patterns identified above.

There is a category in the text, maybe we can use it to filter out some of the noise. For example, if the category is "news", we can prioritize the full news article over the archive summary.