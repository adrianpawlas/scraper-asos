#!/usr/bin/env python3
"""
ASOS Multi-Category Scraper
Scrapes products from ALL ASOS categories using extracted category IDs
"""

import asyncio
import aiohttp
import json
import os
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from supabase import create_client, Client
from tqdm import tqdm
from fake_useragent import UserAgent
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asos_multi_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ASOSMultiCategoryScraper:
    def __init__(self):
        # Supabase configuration
        self.supabase_url = "https://yqawmzggcgpeyaaynrjk.supabase.co"
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4"

        # Initialize Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

        # Initialize embedding model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load CLIP model for image embeddings
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()

        # HTTP session setup
        self.ua = UserAgent()
        self.session_headers = {
            'User-Agent': self.ua.random,
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        # Rate limiting
        self.request_delay = 2.0  # seconds between requests (more conservative)
        self.last_request_time = 0

        # Thread pool for image processing
        self.executor = ThreadPoolExecutor(max_workers=3)  # Reduced for stability

        # Load category data
        self.categories = self.load_categories()
        self.category_ids = self.load_category_ids()

        logger.info(f"Loaded {len(self.categories)} categories and {len(self.category_ids)} category IDs")

    def load_categories(self) -> List[Dict]:
        """Load category information from extracted data"""
        try:
            with open('asos_categories.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Category data not found. Run extract_categories.py first")
            return []

    def load_category_ids(self) -> List[str]:
        """Load category IDs for API calls"""
        try:
            with open('asos_category_ids.txt', 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.warning("Category IDs file not found. Run extract_categories.py first")
            return []

    async def make_request(self, url: str, session: aiohttp.ClientSession, timeout: int = 15) -> Optional[Dict]:
        """Make HTTP request with rate limiting and error handling"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)

        self.last_request_time = time.time()

        try:
            async with session.get(url, headers=self.session_headers, timeout=timeout) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Request failed with status {response.status}: {url}")
                    return None
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
            return None

    def detect_gender(self, product_data: Dict, category_info: Optional[Dict] = None) -> str:
        """Detect gender from product data and category"""
        name = product_data.get('name', '').lower()
        url = product_data.get('url', '').lower()

        # Check category first if available
        if category_info:
            cat_url = category_info.get('web_url', '').lower()
            if '/men/' in cat_url:
                return 'MAN'
            elif '/women/' in cat_url:
                return 'WOMAN'

        # Check for gender indicators in name or URL
        if any(word in name or word in url for word in ['men', 'man', 'male', 'mens', 'guy', 'boy']):
            return 'MAN'
        elif any(word in name or word in url for word in ['women', 'woman', 'female', 'womens', 'girl', 'lady']):
            return 'WOMAN'

        # Default to WOMAN if uncertain (ASOS has more women's products)
        return 'WOMAN'

    def process_price(self, price_data: Dict) -> tuple:
        """Extract price and currency from price data"""
        current_price = price_data.get('current', {})
        value = current_price.get('value')
        currency = price_data.get('currency', 'GBP')

        # Convert GBP to USD (approximate)
        if currency == 'GBP' and value:
            value = round(value * 1.27, 2)  # Current GBP to USD rate
            currency = 'USD'

        return value, currency

    async def download_and_process_image(self, image_url: str) -> Optional[np.ndarray]:
        """Download image and create embedding"""
        try:
            # Ensure full URL
            if not image_url.startswith('http'):
                image_url = f"https://{image_url}"

            # Download image with longer timeout
            response = requests.get(image_url, headers=self.session_headers, timeout=20)
            response.raise_for_status()

            # Open image
            image = Image.open(BytesIO(response.content)).convert('RGB')

            # Process for embedding
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.vision_model(**inputs)
                embedding = outputs.pooler_output.cpu().numpy().flatten()

            # Ensure 768 dimensions
            if len(embedding) > 768:
                embedding = embedding[:768]
            elif len(embedding) < 768:
                embedding = np.pad(embedding, (0, 768 - len(embedding)))

            return embedding

        except Exception as e:
            logger.error(f"Error processing image {image_url}: {e}")
            return None

    def create_metadata(self, product_data: Dict, category_info: Optional[Dict] = None) -> str:
        """Create metadata JSON string from product data"""
        metadata = {
            'product_code': product_data.get('productCode'),
            'colour': product_data.get('colour'),
            'colour_way_id': product_data.get('colourWayId'),
            'brand_name': product_data.get('brandName'),
            'has_variant_colours': product_data.get('hasVariantColours'),
            'product_type': product_data.get('productType'),
            'additional_image_urls': product_data.get('additionalImageUrls', []),
            'is_selling_fast': product_data.get('isSellingFast'),
            'is_promotion': product_data.get('isPromotion'),
            'facets': product_data.get('facetGroupings', []),
            'category_info': category_info
        }
        return json.dumps(metadata)

    def prepare_product_for_db(self, product_data: Dict, embedding: Optional[np.ndarray], category_info: Optional[Dict] = None) -> Dict:
        """Prepare product data for database insertion"""
        price_value, currency = self.process_price(product_data.get('price', {}))

        # Create full URLs
        base_url = "https://www.asos.com"
        product_url = f"{base_url}/{product_data['url']}" if 'url' in product_data else None

        image_url = product_data.get('imageUrl')
        if image_url and not image_url.startswith('http'):
            image_url = f"https://{image_url}"

        product = {
            'id': str(product_data['id']),
            'source': 'scraper',
            'product_url': product_url,
            'affiliate_url': None,
            'image_url': image_url,
            'brand': 'ASOS',
            'title': product_data.get('name', ''),
            'description': None,
            'category': category_info.get('title') if category_info else None,
            'gender': self.detect_gender(product_data, category_info),
            'price': price_value,
            'currency': currency,
            'search_tsv': None,
            'created_at': datetime.now().isoformat(),
            'metadata': self.create_metadata(product_data, category_info),
            'size': None,
            'second_hand': False,
            'embedding': embedding.tolist() if embedding is not None else None,
            'country': 'US',
            'compressed_image_url': None,
            'tags': None,
            'search_vector': None
        }

        return product

    async def fetch_category_products(self, session: aiohttp.ClientSession, category_id: str, category_info: Optional[Dict] = None, offset: int = 0, limit: int = 50) -> List[Dict]:
        """Fetch products from a specific category"""
        # ASOS search API with category filter
        params = {
            'channel': 'mobile-web',
            'country': 'US',
            'currency': 'USD',
            'keyStoreDataversion': '9821',
            'lang': 'en-US',
            'limit': limit,
            'offset': offset,
            'store': 'US',
            'rowlength': 2,
            'cid': category_id  # Category ID filter
        }

        # Build URL
        url = "https://www.asos.com/api/product/search/v2/categories"
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{url}?{query_string}"

        logger.info(f"Fetching category {category_id} ({category_info.get('title', 'Unknown') if category_info else 'Unknown'}) - offset {offset}")

        data = await self.make_request(full_url, session)

        if not data or 'products' not in data:
            return []

        products = data.get('products', [])
        processed_products = []

        # Process products with embeddings
        for product in products:
            # Download image and create embedding
            image_url = product.get('imageUrl')
            embedding = None

            if image_url:
                embedding = await self.download_and_process_image(image_url)

            # Prepare product for database
            db_product = self.prepare_product_for_db(product, embedding, category_info)
            processed_products.append(db_product)

        return processed_products

    async def scrape_category(self, category_id: str, category_info: Optional[Dict] = None) -> List[Dict]:
        """Scrape all products from a single category"""
        all_products = []
        offset = 0
        limit = 50  # Smaller batch size for stability
        max_pages = 10  # Limit pages per category to avoid infinite loops

        async with aiohttp.ClientSession() as session:
            for page in range(max_pages):
                batch = await self.fetch_category_products(session, category_id, category_info, offset, limit)

                if not batch:
                    break

                all_products.extend(batch)
                logger.info(f"Category {category_id}: Got {len(batch)} products (total: {len(all_products)})")

                # Check if we got less than requested (end of category)
                if len(batch) < limit:
                    break

                offset += limit

                # Longer delay between category pages
                await asyncio.sleep(3)

        return all_products

    async def scrape_all_categories(self, max_categories: Optional[int] = None) -> List[Dict]:
        """Scrape products from multiple categories"""
        all_products = []

        # Use first N categories for testing if specified
        categories_to_scrape = self.category_ids[:max_categories] if max_categories else self.category_ids

        logger.info(f"Starting to scrape {len(categories_to_scrape)} categories")

        for i, category_id in enumerate(categories_to_scrape):
            # Find category info
            category_info = None
            for cat in self.categories:
                if cat['category_id'] == category_id:
                    category_info = cat
                    break

            logger.info(f"[{i+1}/{len(categories_to_scrape)}] Scraping category {category_id}: {category_info.get('title', 'Unknown') if category_info else 'Unknown'}")

            try:
                category_products = await self.scrape_category(category_id, category_info)
                all_products.extend(category_products)

                logger.info(f"Category {category_id} completed: {len(category_products)} products")

                # Save progress every 10 categories
                if (i + 1) % 10 == 0:
                    await self.save_progress_checkpoint(all_products, i + 1)

                # Longer delay between categories
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Failed to scrape category {category_id}: {e}")
                continue

        return all_products

    async def save_progress_checkpoint(self, products: List[Dict], categories_completed: int):
        """Save progress to avoid losing work"""
        try:
            checkpoint_data = {
                'categories_completed': categories_completed,
                'total_products': len(products),
                'timestamp': datetime.now().isoformat(),
                'sample_products': products[:5]  # Save first 5 products as sample
            }

            with open(f'checkpoint_{categories_completed}_categories.json', 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Progress checkpoint saved: {categories_completed} categories, {len(products)} products")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    async def save_to_supabase(self, products: List[Dict]) -> int:
        """Save products to Supabase database"""
        success_count = 0

        # First, try a simple test query
        try:
            test_result = self.supabase.table('products').select('count').limit(1).execute()
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            logger.warning("Continuing without database save...")
            return 0

        for product in tqdm(products, desc="Saving to database"):
            try:
                # Try upsert first
                result = self.supabase.table('products').upsert(
                    product,
                    on_conflict='source,product_url'
                ).execute()

                success_count += 1

            except Exception as e:
                logger.warning(f"Upsert failed for product {product.get('id')}, trying insert: {e}")

                try:
                    # Try regular insert
                    result = self.supabase.table('products').insert(product).execute()
                    success_count += 1

                except Exception as e2:
                    logger.error(f"Insert also failed for product {product.get('id')}: {e2}")

                    # Save failed products to a file for later retry
                    try:
                        with open('failed_products_multi.json', 'a', encoding='utf-8') as f:
                            json.dump(product, f, ensure_ascii=False)
                            f.write('\n')
                    except Exception as e3:
                        logger.error(f"Could not save failed product to file: {e3}")

                    continue

        return success_count

    async def run_scraper(self, max_categories: Optional[int] = None):
        """Main scraper execution"""
        logger.info("Starting ASOS Multi-Category Scraper")

        try:
            # Scrape products from categories
            logger.info("Starting multi-category scraping...")
            products = await self.scrape_all_categories(max_categories)

            if not products:
                logger.warning("No products found! ASOS may be blocking requests.")
                logger.info("Try using the sample data file or check network connectivity.")
                return

            logger.info(f"Successfully processed {len(products)} products from {len(self.category_ids[:max_categories] if max_categories else self.category_ids)} categories")

            # Show sample of what we got
            if products:
                sample = products[0]
                logger.info(f"Sample product: {sample['title'][:50]}...")
                logger.info(f"  Gender: {sample['gender']}")
                logger.info(f"  Price: {sample['price']} {sample['currency']}")
                logger.info(f"  Category: {sample.get('category', 'N/A')}")
                logger.info(f"  Has embedding: {sample['embedding'] is not None}")

            # Save to database
            logger.info("Saving products to Supabase...")
            saved_count = await self.save_to_supabase(products)

            if saved_count > 0:
                logger.info(f"Successfully saved {saved_count} products to database")
            else:
                logger.warning("No products saved to database. Check database configuration.")
                logger.info(f"Failed products saved to 'failed_products_multi.json' for later retry")

            # Save successful products to JSON file as backup
            try:
                with open('successful_products_multi.json', 'w', encoding='utf-8') as f:
                    json.dump(products, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(products)} products to 'successful_products_multi.json'")
            except Exception as e:
                logger.error(f"Could not save products to JSON file: {e}")

        except Exception as e:
            logger.error(f"Scraper failed: {e}")
            raise
        finally:
            # Cleanup
            self.executor.shutdown()

async def main():
    """Main entry point"""
    import sys

    # Allow specifying max categories as command line argument
    max_categories = None
    if len(sys.argv) > 1:
        try:
            max_categories = int(sys.argv[1])
        except ValueError:
            pass

    scraper = ASOSMultiCategoryScraper()
    await scraper.run_scraper(max_categories)

if __name__ == "__main__":
    asyncio.run(main())

