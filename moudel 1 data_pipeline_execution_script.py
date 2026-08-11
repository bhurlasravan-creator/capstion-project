import logging
import re
import sqlite3
import urllib.parse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

BASE_URL = "http://books.toscrape.com/"
INR_FIXED_RATE = 105.50  # Required fixed baseline: 1 GBP = 105.50 INR
DB_PATH = "catalog_benchmark.db"

# Cache for detail page category lookups to minimize HTTP requests
_CATEGORY_CACHE: Dict[str, str] = {}

# Mapping string star ratings to integers
RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

# Raw text snippet input provided by user for additional catalog results
USER_PROVIDED_RAW_RESULTS = """
Warning! This is a demo website for web scraping purposes. Prices and ratings here were randomly assigned and have no real meaning.

Louisa: The Extraordinary Life ...
£16.85
In stock

Setting the World on ...
£21.15
In stock

The Faith of Christopher ...
£39.55
In stock

Benjamin Franklin: An American ...
£48.19
In stock

Chasing Lincoln's Killer ...
£15.55
In stock

It's Only the Himalayas
£45.17
In stock

Full Moon over Noah’s ...
£49.43
In stock

See America: A Celebration ...
£48.87
In stock

Vagabonding: An Uncommon Guide ...
£36.94
In stock

Under the Tuscan Sun
£37.33
In stock

A Summer In Europe
£44.34
In stock

The Great Railway Bazaar
£30.54
In stock

A Year in Provence ...
£56.88
In stock

The Road to Little ...
£23.21
In stock

Neither Here nor There: ...
£38.95
In stock

1,000 Places to See ...
£26.08
In stock
"""


def parse_pasted_text_results(raw_text: str) -> List[dict]:
    """
    Parses unstructured or copy-pasted book text snippets (containing Title, Price in GBP,
    and Availability) into structured dictionary records for the data pipeline.
    Smartly infers whether a title belongs to 'Travel' or 'Biography'.
    """
    logging.info("Parsing user-provided raw text book results with smart category mapping...")
    parsed_records = []
    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]

    # Keywords to assign 'Travel' category automatically
    travel_keywords = [
        "himalayas", "noah", "america", "vagabonding", "tuscan", "europe", 
        "railway", "provence", "road", "places", "travel", "there"
    ]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # Ignore disclaimer / header text
        if "Warning!" in line or "demo website" in line:
            i += 1
            continue
            
        # Check if line looks like a title followed by price on next line
        if i + 1 < len(lines) and re.search(r"£\d+\.\d{2}", lines[i + 1]):
            title = line
            price_raw = lines[i + 1]
            availability_raw = lines[i + 2] if (i + 2 < len(lines) and "stock" in lines[i + 2].lower()) else "In stock"
            
            # Infer category based on title contents
            if any(kw in title.lower() for kw in travel_keywords):
                inferred_category = "Travel"
            else:
                inferred_category = "Biography"

            parsed_records.append({
                "title": title,
                "price_raw": price_raw,
                "star_rating_raw": "Four",  # Default baseline rating for snippet items if unspecified
                "availability_raw": availability_raw,
                "category": inferred_category,
            })
            i += 3
        else:
            i += 1

    logging.info(f"Successfully extracted {len(parsed_records)} additional records from raw text.")
    return parsed_records


def scrape_books_data(target_count: int = 60, max_pages: int = 5) -> pd.DataFrame:
    """
    Scrapes catalog pricing and availability data from books.toscrape.com across multiple paginated pages.
    Captures title, price_raw, star_rating, availability_raw, and category.
    """
    logging.info("Starting web scraping from books.toscrape.com...")
    records = []
    page_url = "catalogue/page-1.html"

    session = requests.Session()
    session.headers.update({"User-Agent": "Zepto-Catalog-DataPipeline/1.0"})

    pages_scraped = 0

    while page_url and pages_scraped < max_pages and len(records) < target_count:
        full_url = urllib.parse.urljoin(BASE_URL, page_url)
        logging.info(f"Fetching page {pages_scraped + 1}: {full_url}")

        try:
            response = session.get(full_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch {full_url}: {e}")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        products = soup.select("article.product_pod")

        for product in products:
            # 1. Title
            title_elem = product.select_one("h3 a")
            title = title_elem.get("title") if title_elem and title_elem.has_attr("title") else (
                title_elem.text.strip() if title_elem else None
            )

            # 2. Price (GBP)
            price_elem = product.select_one(".price_color")
            price_raw = price_elem.text.strip() if price_elem else None

            # 3. Star Rating
            rating_elem = product.select_one("p.star-rating")
            rating_class = None
            if rating_elem:
                classes = rating_elem.get("class", [])
                rating_class = next((c for c in classes if c != "star-rating"), None)

            # 4. Availability
            availability_elem = product.select_one(".availability")
            availability_raw = availability_elem.text.strip() if availability_elem else None

            # 5. Category lookup by fetching detail page link
            book_detail_rel = title_elem.get("href") if title_elem else ""
            detail_full_url = urllib.parse.urljoin(full_url, book_detail_rel)
            category_name = fetch_book_category(session, detail_full_url)

            records.append({
                "title": title,
                "price_raw": price_raw,
                "star_rating_raw": rating_class,
                "availability_raw": availability_raw,
                "category": category_name,
            })

        pages_scraped += 1

        # Check for next page
        next_button = soup.select_one("li.next a")
        if next_button:
            next_href = next_button.get("href")
            if page_url.startswith("catalogue/"):
                page_url = f"catalogue/{next_href}"
            else:
                page_url = next_href
        else:
            page_url = None

    # Parse and append user-provided pasted book results
    pasted_records = parse_pasted_text_results(USER_PROVIDED_RAW_RESULTS)
    records.extend(pasted_records)

    logging.info(f"Scraping & snippet parsing complete. Total records gathered: {len(records)}")
    return pd.DataFrame(records)


def fetch_book_category(session: requests.Session, detail_url: str) -> str:
    """Helper function to extract category name from detail page breadcrumbs with in-memory caching."""
    if detail_url in _CATEGORY_CACHE:
        return _CATEGORY_CACHE[detail_url]

    try:
        resp = session.get(detail_url, timeout=5)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            breadcrumbs = soup.select("ul.breadcrumb li a")
            if len(breadcrumbs) >= 3:
                cat_name = breadcrumbs[2].text.strip()
                _CATEGORY_CACHE[detail_url] = cat_name
                return cat_name
    except Exception as e:
        logging.debug(f"Could not fetch category for {detail_url}: {e}")

    fallback = "Default Category"
    _CATEGORY_CACHE[detail_url] = fallback
    return fallback


def clean_and_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw web-scraped fields into proper typed columns:
    - Strips currency symbol from price -> price_gbp (float)
    - Maps text rating -> rating (int 1-5)
    - Parses availability -> in_stock (boolean/int 0 or 1)
    - Applies median imputation for missing numeric fields & drops critical invalid rows
    - Converts price_gbp -> price_inr using fixed rate (1 GBP = 105.50 INR)
    """
    logging.info("Starting data cleaning and transformation...")
    clean_df = df.copy()

    # Drop rows missing essential text identifier (title)
    initial_count = len(clean_df)
    clean_df.dropna(subset=["title"], inplace=True)
    if len(clean_df) < initial_count:
        logging.info(f"Dropped {initial_count - len(clean_df)} row(s) lacking titles.")

    # 1. Clean Price (GBP)
    def parse_price(val):
        if pd.isna(val):
            return np.nan
        cleaned = re.sub(r"[^\d.]", "", str(val))
        try:
            return float(cleaned)
        except ValueError:
            return np.nan

    clean_df["price_gbp"] = clean_df["price_raw"].apply(parse_price)

    # Median Imputation for missing numeric price_gbp
    if clean_df["price_gbp"].isnull().any():
        median_price = clean_df["price_gbp"].median()
        logging.warning(f"Imputing missing price_gbp values with median: {median_price:.2f}")
        clean_df["price_gbp"].fillna(median_price, inplace=True)

    # 2. Map Star Rating to Integer (1-5)
    clean_df["rating"] = clean_df["star_rating_raw"].map(RATING_MAP)
    if clean_df["rating"].isnull().any():
        median_rating = int(clean_df["rating"].median()) if not np.isnan(clean_df["rating"].median()) else 4
        logging.warning(f"Imputing missing rating values with median: {median_rating}")
        clean_df["rating"].fillna(median_rating, inplace=True)
    clean_df["rating"] = clean_df["rating"].astype(int)

    # 3. Parse Availability into Boolean Integer (1 = True, 0 = False)
    clean_df["in_stock"] = clean_df["availability_raw"].apply(
        lambda x: 1 if isinstance(x, str) and "in stock" in x.lower() else 0
    )

    # 4. Baseline Currency Conversion: price_inr = price_gbp * 105.50
    clean_df["price_inr"] = (clean_df["price_gbp"] * INR_FIXED_RATE).round(2)

    # Select and finalize clean columns
    clean_df = clean_df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]
    logging.info("Data cleaning complete.")
    return clean_df


def create_and_populate_sqlite(df: pd.DataFrame, db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Creates normalized SQLite schema with two tables:
    - categories (category_id, category_name)
    - books (book_id, title, price_gbp, price_inr, rating, in_stock, category_id)
    Inserts cleaned data adhering to FK relationship.
    """
    logging.info(f"Initializing SQLite database at '{db_path}'...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Drop existing tables for fresh rerun
    cursor.execute("DROP TABLE IF EXISTS books;")
    cursor.execute("DROP TABLE IF EXISTS categories;")

    # Create Normalized Tables
    cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories (category_id)
    );
    """)

    # Populate categories table
    unique_categories = sorted(df["category"].unique())
    cursor.executemany(
        "INSERT INTO categories (category_name) VALUES (?);",
        [(cat,) for cat in unique_categories]
    )

    # Fetch mapping of category_name -> category_id
    cursor.execute("SELECT category_name, category_id FROM categories;")
    cat_map = dict(cursor.fetchall())

    # Map category string to FK integer
    df_books = df.copy()
    df_books["category_id"] = df_books["category"].map(cat_map)

    # Populate books table
    books_data = df_books[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]].to_dict("records")
    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (:title, :price_gbp, :price_inr, :rating, :in_stock, :category_id);
    """, books_data)

    conn.commit()
    logging.info("Database successfully populated with normalized schema.")
    return conn


def run_sql_queries(conn: sqlite3.Connection) -> List[Tuple[str, pd.DataFrame]]:
    """
    Executes required 5 SQL queries demonstrating:
    1. SELECT/WHERE + ORDER BY
    2. DISTINCT + LIMIT
    3. IN clause
    4. BETWEEN clause
    5. JOIN clause
    """
    queries = [
        (
            "Query 1: High-Value Available Books (SELECT/WHERE + ORDER BY)",
            """
            SELECT book_id, title, price_gbp, price_inr, rating 
            FROM books 
            WHERE in_stock = 1 AND price_gbp > 30.0 
            ORDER BY price_gbp DESC;
            """
        ),
        (
            "Query 2: Top 10 Distinct Ratings & Prices Sample (DISTINCT + LIMIT)",
            """
            SELECT DISTINCT rating, price_gbp 
            FROM books 
            ORDER BY rating DESC, price_gbp DESC 
            LIMIT 10;
            """
        ),
        (
            "Query 3: Filter Books by Category Subset (IN Clause)",
            """
            SELECT b.book_id, b.title, c.category_name, b.price_inr 
            FROM books b 
            JOIN categories c ON b.category_id = c.category_id 
            WHERE c.category_name IN ('Travel', 'Sequential Art', 'Mystery', 'Music', 'Biography', 'Default Category') 
            ORDER BY c.category_name, b.title;
            """
        ),
        (
            "Query 4: Mid-Tier Priced Books in INR (BETWEEN Clause)",
            """
            SELECT book_id, title, price_inr, rating 
            FROM books 
            WHERE price_inr BETWEEN 2000.0 AND 4000.0 
            ORDER BY price_inr ASC;
            """
        ),
        (
            "Query 5: Category Benchmark Overview (JOIN Clause)",
            """
            SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating 
            FROM books b 
            JOIN categories c ON b.category_id = c.category_id 
            ORDER BY c.category_name, b.rating DESC, b.price_gbp ASC;
            """
        )
    ]

    executed_results = []
    print("\n" + "=" * 80)
    print("EXECUTING SQL QUERIES AGAINST SQLITE DATABASE")
    print("=" * 80)

    for title, query_str in queries:
        df_result = pd.read_sql_query(query_str, conn)
        executed_results.append((title, df_result))
        print(f"\n--- {title} ---")
        print(f"SQL String:\n{query_str.strip()}\n")
        print("Result Sample:")
        print(df_result.head(10).to_string(index=False))
        print(f"Total Rows Returned: {len(df_result)}")
        print("-" * 60)

    return executed_results


def compare_sql_and_pandas_merge(conn: sqlite3.Connection):
    """
    Reads back tables using pd.read_sql and reproduces Query 5 JOIN logic 
    using pd.merge directly on in-memory DataFrames, proving equivalence.
    """
    print("\n" + "=" * 80)
    print("PANDAS vs SQL JOIN EQUIVALENCE VERIFICATION")
    print("=" * 80)

    # Read back raw tables using pd.read_sql
    books_df = pd.read_sql_query("SELECT * FROM books;", conn)
    categories_df = pd.read_sql_query("SELECT * FROM categories;", conn)

    # 1. SQL JOIN execution via read_sql
    sql_join_query = """
    SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating 
    FROM books b 
    JOIN categories c ON b.category_id = c.category_id 
    ORDER BY c.category_name, b.rating DESC, b.price_gbp ASC;
    """
    sql_join_df = pd.read_sql_query(sql_join_query, conn)

    # 2. Pure Pandas pd.merge execution
    pandas_merged_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    # Filter & Sort to align with SQL query format
    pandas_merged_df = pandas_merged_df[
        ["book_id", "title", "category_name", "price_gbp", "price_inr", "rating"]
    ].sort_values(
        by=["category_name", "rating", "price_gbp"],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    # Normalize index for exact comparison
    sql_join_df = sql_join_df.reset_index(drop=True)

    print("\n[SQL Read Result Head]:")
    print(sql_join_df.head(5).to_string(index=False))

    print("\n[Pandas Merge Result Head]:")
    print(pandas_merged_df.head(5).to_string(index=False))

    # Verification check
    is_equivalent = sql_join_df.equals(pandas_merged_df)
    print(f"\nExact Dataframe Equivalence Match (pd.read_sql vs pd.merge): {is_equivalent}")
    assert is_equivalent, "Mismatch found between SQL JOIN and Pandas merge results!"
    print("SUCCESS: Pure Pandas merge output identically matches the SQL JOIN query!")


def main():
    # Step 1: Scrape & Parse Data
    raw_df = scrape_books_data(target_count=60, max_pages=5)
    print(f"\nRaw Dataset Summary: {len(raw_df)} records gathered.")

    # Step 2: Clean & Convert Data
    clean_df = clean_and_transform_data(raw_df)
    print(f"Cleaned Dataset Summary: {len(clean_df)} records ready.")

    # Step 3: Populate SQLite Database
    conn = create_and_populate_sqlite(clean_df, DB_PATH)

    # Step 4: Execute SQL Queries
    run_sql_queries(conn)

    # Step 5: Pandas Merge Comparison
    compare_sql_and_pandas_merge(conn)

    conn.close()
    logging.info("Module 1 Data Pipeline finished successfully.")


if __name__ == "__main__":
    main()