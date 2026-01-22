#!/usr/bin/env python3
"""
ASOS Fashion Scraper
Scrapes all products from ASOS, creates image embeddings, and stores in Supabase
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
        logging.FileHandler('asos_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ASOSScraper:
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
        self.request_delay = 1.0  # seconds between requests
        self.last_request_time = 0

        # Thread pool for image processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def make_request(self, url: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Make HTTP request with rate limiting and error handling"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)

        self.last_request_time = time.time()

        try:
            async with session.get(url, headers=self.session_headers, timeout=30) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Request failed with status {response.status}: {url}")
                    return None
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
            return None

    def detect_gender(self, product_data: Dict) -> str:
        """Detect gender from product data"""
        name = product_data.get('name', '').lower()
        url = product_data.get('url', '').lower()

        # Check for gender indicators in name or URL
        if any(word in name or word in url for word in ['men', 'man', 'male', 'mens', 'guy', 'boy']):
            return 'MAN'
        elif any(word in name or word in url for word in ['women', 'woman', 'female', 'womens', 'girl', 'lady']):
            return 'WOMAN'

        # Check category/facets if available
        facets = product_data.get('facetGroupings', [])
        for facet in facets:
            if facet.get('name', '').lower() in ['gender', 'category']:
                values = facet.get('facetValues', [])
                for value in values:
                    val_name = value.get('name', '').lower()
                    if 'men' in val_name or 'male' in val_name:
                        return 'MAN'
                    elif 'women' in val_name or 'female' in val_name:
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

            # Download image
            response = requests.get(image_url, headers=self.session_headers, timeout=10)
            response.raise_for_status()

            # Open image
            image = Image.open(BytesIO(response.content)).convert('RGB')

            # Process for embedding
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.vision_model(**inputs)
                # Use the pooled output from the vision model
                embedding = outputs.pooler_output.cpu().numpy().flatten()

            # Ensure 768 dimensions (truncate or pad if necessary)
            if len(embedding) > 768:
                embedding = embedding[:768]
            elif len(embedding) < 768:
                embedding = np.pad(embedding, (0, 768 - len(embedding)))

            return embedding

        except Exception as e:
            logger.error(f"Error processing image {image_url}: {e}")
            return None

    def create_metadata(self, product_data: Dict) -> str:
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
            'facets': product_data.get('facetGroupings', [])
        }
        return json.dumps(metadata)

    def prepare_product_for_db(self, product_data: Dict, embedding: Optional[np.ndarray]) -> Dict:
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
            'affiliate_url': None,  # Can be set later if needed
            'image_url': image_url,
            'brand': 'ASOS',
            'title': product_data.get('name', ''),
            'description': None,  # ASOS doesn't provide descriptions in search API
            'category': None,  # Would need additional API calls
            'gender': self.detect_gender(product_data),
            'price': price_value,
            'currency': currency,
            'search_tsv': None,  # Will be computed by database
            'created_at': datetime.now().isoformat(),
            'metadata': self.create_metadata(product_data),
            'size': None,  # Size variants would need additional processing
            'second_hand': False,
            'embedding': embedding.tolist() if embedding is not None else None,
            'country': 'US',  # Default to US
            'compressed_image_url': None,  # Can be set later
            'tags': None,  # Can be extracted from facets later
            'search_vector': None  # Will be computed by database
        }

        return product

    async def fetch_products_page(self, session: aiohttp.ClientSession, offset: int = 0, limit: int = 72) -> Optional[Dict]:
        """Fetch a page of products from ASOS search API"""
        # ASOS search API parameters
        params = {
            'channel': 'mobile-web',
            'country': 'US',
            'currency': 'USD',
            'keyStoreDataversion': '9821',
            'lang': 'en-US',
            'limit': limit,
            'offset': offset,
            'store': 'US',
            'rowlength': 2
        }

        # Build URL - this is the main search endpoint
        url = "https://www.asos.com/api/product/search/v2/categories"

        # Add query parameters
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{url}?{query_string}"

        logger.info(f"Fetching products with offset {offset}")
        return await self.make_request(full_url, session)

    async def scrape_all_products(self) -> List[Dict]:
        """Scrape all products from ASOS"""
        all_products = []

        # First, try to load from sample data file if available
        try:
            import json
            with open('3.txt', 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
                sample_products = sample_data.get('products', [])
                logger.info(f"Loaded {len(sample_products)} products from sample data")

                # Process sample products
                for product in tqdm(sample_products, desc="Processing sample products"):
                    # Download image and create embedding
                    image_url = product.get('imageUrl')
                    embedding = None

                    if image_url:
                        embedding = await self.download_and_process_image(image_url)

                    # Prepare product for database
                    db_product = self.prepare_product_for_db(product, embedding)
                    all_products.append(db_product)

        except FileNotFoundError:
            logger.warning("Sample data file not found, trying API...")

        # Then try API (currently blocked by ASOS)
        if not all_products:
            logger.info("Attempting to fetch from ASOS API...")
            offset = 0
            limit = 72

            async with aiohttp.ClientSession() as session:
                while True:
                    # Fetch page
                    data = await self.fetch_products_page(session, offset, limit)

                    if not data or 'products' not in data:
                        logger.info(f"No more products found at offset {offset}")
                        break

                    products = data['products']
                    if not products:
                        break

                    logger.info(f"Fetched {len(products)} products at offset {offset}")

                    # Process products
                    processed_products = []
                    for product in tqdm(products, desc=f"Processing products (offset {offset})"):
                        image_url = product.get('imageUrl')
                        embedding = None

                        if image_url:
                            embedding = await self.download_and_process_image(image_url)

                        db_product = self.prepare_product_for_db(product, embedding)
                        processed_products.append(db_product)

                    all_products.extend(processed_products)

                    if len(products) < limit:
                        break

                    offset += limit
                    await asyncio.sleep(2)

        return all_products

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
                        with open('failed_products.json', 'a', encoding='utf-8') as f:
                            json.dump(product, f, ensure_ascii=False)
                            f.write('\n')
                    except Exception as e3:
                        logger.error(f"Could not save failed product to file: {e3}")

                    continue

        return success_count

    async def run_scraper(self):
        """Main scraper execution"""
        logger.info("Starting ASOS scraper")

        try:
            # Scrape all products
            logger.info("Fetching products from ASOS...")
            products = await self.scrape_all_products()

            if not products:
                logger.warning("No products found! ASOS may be blocking requests.")
                logger.info("Try using the sample data file (3.txt) or check network connectivity.")
                return

            logger.info(f"Successfully processed {len(products)} products")

            # Show sample of what we got
            if products:
                sample = products[0]
                logger.info(f"Sample product: {sample['title'][:50]}...")
                logger.info(f"  Gender: {sample['gender']}")
                logger.info(f"  Price: {sample['price']} {sample['currency']}")
                logger.info(f"  Has embedding: {sample['embedding'] is not None}")

            # Save to database
            logger.info("Saving products to Supabase...")
            saved_count = await self.save_to_supabase(products)

            if saved_count > 0:
                logger.info(f"Successfully saved {saved_count} products to database")
            else:
                logger.warning("No products saved to database. Check database configuration.")
                logger.info(f"Failed products saved to 'failed_products.json' for later retry")

            # Save successful products to JSON file as backup
            try:
                successful_products = products[:saved_count] if saved_count > 0 else products
                with open('successful_products.json', 'w', encoding='utf-8') as f:
                    json.dump(successful_products, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(successful_products)} products to 'successful_products.json'")
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
    scraper = ASOSScraper()
    await scraper.run_scraper()

if __name__ == "__main__":
    asyncio.run(main())
