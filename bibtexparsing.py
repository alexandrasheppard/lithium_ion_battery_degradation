import bibtexparser
from datetime import datetime


def format_authors(authors):
    # Format authors in "Last First, Last First, et al." format if there are more than 3 authors
    author_list = [a.strip() for a in authors.split(' and ')]
    if len(author_list) > 3:
        return f"{', '.join(author_list[:3])}, et al."
    return ', '.join(author_list)


def format_entry(entry, index):
    # Extract fields from the entry, using placeholders if fields are missing
    author = format_authors(entry.get("author", "Unknown Author"))
    title = entry.get("title", "No Title").replace('{', '').replace('}', '')
    journal = entry.get("journal", entry.get("journaltitle", "Unknown Journal"))
    year = entry.get("year", "Unknown Year")
    month = entry.get("month", "")
    volume = entry.get("volume", "")
    number = entry.get("number", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")
    url = entry.get("url", "")
    urldate = entry.get("urldate", "")

    # Format journal info with volume, issue, and pages
    journal_info = f"{journal} {year};{volume}"
    if number:
        journal_info += f"({number})"
    if pages:
        journal_info += f":{pages}"

    # Format DOI and URL with access date
    doi_part = f"http://dx.doi.org/{doi}" if doi else ""
    url_part = f", URL {url}" if url else ""
    urldate_part = (
        f". [Accessed {datetime.strptime(urldate, '%Y-%m-%d').strftime('%d %B %Y')}]" if urldate else ""
    )

    # Combine all parts into the formatted reference
    return f"[{index}] {author}. {title}. {journal_info}. {doi_part}{url_part}{urldate_part}."


def process_bib_file(bib_filename):
    with open(bib_filename, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    # Process each entry and create formatted references
    formatted_entries = []
    for i, entry in enumerate(bib_database.entries, 1):
        formatted_entries.append(format_entry(entry, i))

    return "\n".join(formatted_entries)


# Specify the .bib file path
bib_filename = r"M:\Downloads\References.bib"
formatted_references = process_bib_file(bib_filename)
print(formatted_references)
