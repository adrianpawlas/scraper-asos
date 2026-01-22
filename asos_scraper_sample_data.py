#!/usr/bin/env python3
"""
ASOS Scraper using Sample Data Files
Fallback when browser automation is blocked
Uses the sample API responses from 1.txt, 2.txt, 3.txt
"""

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asos_sample_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ASOSSampleDataScraper:
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

        # Thread pool for image processing
        self.executor = ThreadPoolExecutor(max_workers=3)

    def detect_gender(self, product_data: Dict, category_info: Optional[Dict] = None) -> str:
        """Detect gender from product data"""
        name = product_data.get('name', '').lower()
        url = product_data.get('url', '').lower()

        # Check for gender indicators
        if any(word in name or word in url for word in ['men', 'man', 'male', 'mens', 'guy', 'boy']):
            return 'MAN'
        elif any(word in name or word in url for word in ['women', 'woman', 'female', 'womens', 'girl', 'lady']):
            return 'WOMAN'

        return 'WOMAN'  # Default

    def process_price(self, price_data: Dict) -> tuple:
        """Extract price and currency"""
        current_price = price_data.get('current', {})
        value = current_price.get('value')
        currency = price_data.get('currency', 'GBP')

        # Convert GBP to USD
        if currency == 'GBP' and value:
            value = round(value * 1.27, 2)
            currency = 'USD'

        return value, currency

    async def download_and_process_image(self, image_url: str) -> Optional[np.ndarray]:
        """Download image and create embedding"""
        try:
            if not image_url.startswith('http'):
                image_url = f"https://{image_url}"

            response = requests.get(image_url, timeout=20, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert('RGB')

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

    def create_metadata(self, product_data: Dict) -> str:
        """Create metadata JSON"""
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
            'scraped_via': 'sample_data_files'
        }
        return json.dumps(metadata)

    def prepare_product_for_db(self, product_data: Dict, embedding: Optional[np.ndarray]) -> Dict:
        """Prepare product for database"""
        price_value, currency = self.process_price(product_data.get('price', {}))

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
            'category': product_data.get('categoryName', 'New in: Today'),
            'gender': self.detect_gender(product_data),
            'price': price_value,
            'currency': currency,
            'search_tsv': None,
            'created_at': datetime.now().isoformat(),
            'metadata': self.create_metadata(product_data),
            'size': None,
            'second_hand': False,
            'embedding': embedding.tolist() if embedding is not None else None,
            'country': 'US',
            'compressed_image_url': None,
            'tags': None,
            'search_vector': None
        }

        return product

    def load_sample_data(self) -> List[Dict]:
        """Load products from 3.txt sample data file"""
        products = []

        try:
            with open('3.txt', 'r', encoding='utf-8') as f:
                data = json.load(f)

            products_data = data.get('products', [])
            logger.info(f"Loaded {len(products_data)} products from 3.txt")

            return products_data

        except FileNotFoundError:
            logger.error("3.txt file not found!")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing 3.txt: {e}")
            return []

    async def process_sample_products(self) -> List[Dict]:
        """Process products from sample data"""
        products_data = self.load_sample_data()

        if not products_data:
            logger.warning("No products found in sample data")
            return []

        processed_products = []

        logger.info(f"Processing {len(products_data)} products from sample data...")

        for product_data in tqdm(products_data, desc="Processing products"):
            # Download image and create embedding
            image_url = product_data.get('imageUrl')
            embedding = None

            if image_url:
                embedding = await self.download_and_process_image(image_url)

            # Prepare product for database
            db_product = self.prepare_product_for_db(product_data, embedding)
            processed_products.append(db_product)

        return processed_products

    async def save_to_supabase(self, products: List[Dict]) -> int:
        """Save products to Supabase"""
        success_count = 0

        try:
            test_result = self.supabase.table('products').select('count').limit(1).execute()
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return 0

        for product in tqdm(products, desc="Saving to database"):
            try:
                result = self.supabase.table('products').upsert(
                    product,
                    on_conflict='source,product_url'
                ).execute()
                success_count += 1

            except Exception as e:
                logger.warning(f"Upsert failed for product {product.get('id')}, trying insert: {e}")

                try:
                    result = self.supabase.table('products').insert(product).execute()
                    success_count += 1

                except Exception as e2:
                    logger.error(f"Insert also failed for product {product.get('id')}: {e2}")
                    continue

        return success_count

    async def run_scraper(self):
        """Main scraper execution"""
        logger.info("Starting ASOS Sample Data Scraper")

        try:
            # Process sample products
            logger.info("Loading and processing sample data...")
            products = await self.process_sample_products()

            if not products:
                logger.warning("No products processed!")
                return

            logger.info(f"Successfully processed {len(products)} products")

            # Show sample
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
                logger.warning("No products saved to database")

            # Save backup
            try:
                with open('successful_products_sample.json', 'w', encoding='utf-8') as f:
                    json.dump(products, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(products)} products to 'successful_products_sample.json'")
            except Exception as e:
                logger.error(f"Could not save products to JSON file: {e}")

        except Exception as e:
            logger.error(f"Scraper failed: {e}")
            raise
        finally:
            self.executor.shutdown()

async def main():
    """Main entry point"""
    scraper = ASOSSampleDataScraper()
    await scraper.run_scraper()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())