#!/usr/bin/env python3
"""
ASOS Browser-Based Scraper
Uses Playwright to scrape ASOS products directly from web pages
Bypasses API blocking by using real browser automation
"""

import asyncio
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
from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asos_browser_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ASOSBrowserScraper:
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
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced for stability

        # Load category data
        self.categories = self.load_categories()
        self.category_ids = self.load_category_ids()

        logger.info(f"Loaded {len(self.categories)} categories and {len(self.category_ids)} category IDs")

        # Browser settings
        self.browser = None
        self.context = None

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

    async def init_browser(self):
        """Initialize Playwright browser"""
        if self.browser is None:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--single-process',  # <- this one doesn't work in Windows
                    '--disable-gpu'
                ]
            )
            self.context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
                permissions=['geolocation'],
                geolocation={'latitude': 40.7128, 'longitude': -74.0060},  # New York coordinates
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Cache-Control': 'max-age=0'
                }
            )

    async def close_browser(self):
        """Close browser and cleanup"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()

    async def scrape_category_page(self, category_url: str, category_info: Optional[Dict] = None) -> List[Dict]:
        """Scrape products from a category page using browser automation"""
        await self.init_browser()

        products = []
        page = await self.context.new_page()

        try:
            logger.info(f"Loading category page: {category_url}")

            # Monitor network requests to see if products are loaded via API
            api_calls = []

            def log_request(request):
                if 'api' in request.url.lower() or 'product' in request.url.lower():
                    api_calls.append({
                        'url': request.url,
                        'method': request.method,
                        'headers': dict(request.headers)
                    })
                    logger.info(f"API call: {request.method} {request.url}")

            page.on('request', log_request)

            await page.goto(category_url, wait_until='networkidle', timeout=30000)

            # Wait for initial page load
            await page.wait_for_timeout(5000)

            # Load additional products using pagination
            logger.info("Loading additional product pages...")
            await self.scroll_and_wait(page)

            # Additional wait for any remaining AJAX content
            await page.wait_for_timeout(3000)

            # Debug: Check page structure
            page_info = await page.evaluate("""
                () => {
                    const info = {
                        title: document.title,
                        url: window.location.href,
                        productTileCount: document.querySelectorAll('[data-auto-id="productTile"]').length,
                        productCardCount: document.querySelectorAll('.product-card').length,
                        productItemCount: document.querySelectorAll('.product-item').length,
                        productLinks: document.querySelectorAll('a[href*="/prd/"]').length,
                        allDivs: document.querySelectorAll('div').length,
                        bodyText: document.body.textContent.substring(0, 500)
                    };
                    return info;
                }
            """)

            logger.info(f"Page info: {page_info}")

            # Extract product data from all loaded pages
            product_data = await page.evaluate("""
                () => {
                    const products = [];

                    // Try multiple selectors for ASOS products
                    const selectors = [
                        '[data-auto-id="productTile"]',
                        '[data-testid="productTile"]',
                        '.productTile',
                        '.product-card',
                        '.product-item',
                        '[data-product]',
                        '.product'
                    ];

                    let productElements = [];
                    for (const selector of selectors) {
                        productElements = document.querySelectorAll(selector);
                        if (productElements.length > 0) {
                            console.log(`Found ${productElements.length} products with selector: ${selector}`);
                            break;
                        }
                    }

                    // If no products found with specific selectors, try to find all links to products
                    if (productElements.length === 0) {
                        const allLinks = document.querySelectorAll('a[href*="/prd/"]');
                        console.log(`Found ${allLinks.length} product links on page`);

                        // Group links by product ID
                        const productMap = new Map();

                        allLinks.forEach(link => {
                            const href = link.href;
                            const match = href.match(/\/prd\/(\d+)/);
                            if (match) {
                                const productId = match[1];
                                if (!productMap.has(productId)) {
                                    productMap.set(productId, {
                                        link: link,
                                        href: href
                                    });
                                }
                            }
                        });

                        // Convert map to array and create product elements
                        productElements = Array.from(productMap.values()).map(item => item.link.parentElement || item.link);
                        console.log(`Grouped into ${productElements.length} unique products`);
                    }

                    console.log(`Processing ${productElements.length} product elements`);

                    productElements.forEach((element, index) => {
                        try {
                            // Extract product information
                            const link = element.querySelector('a[href*="/prd/"]') || element.closest('a[href*="/prd/"]') || element;
                            const img = element.querySelector('img') || element.querySelector('[data-src]');
                            const priceElement = element.querySelector('[data-auto-id*="price"], .price, .product-price, [class*="price"]');
                            const nameElement = element.querySelector('[data-auto-id*="title"], [data-auto-id*="name"], .product-name, .title, h3, h4');

                            if (link && link.href && link.href.includes('/prd/')) {
                                const productId = link.href.match(/\/prd\/(\d+)/)?.[1] || `temp_${index}`;

                                const product = {
                                    id: productId,
                                    name: nameElement?.textContent?.trim() || link?.textContent?.trim() || `Product ${productId}`,
                                    url: link.href,
                                    imageUrl: img?.src || img?.getAttribute('data-src') || img?.getAttribute('srcset')?.split(',')[0]?.split(' ')[0] || '',
                                    price: {
                                        current: {
                                            value: null,
                                            text: priceElement?.textContent?.trim() || ''
                                        }
                                    },
                                    brandName: 'ASOS',
                                    colour: '',
                                    colourWayId: null,
                                    hasVariantColours: false,
                                    productCode: null,
                                    productType: 'Product'
                                };

                                // Try to extract price value
                                const priceText = product.price.current.text;
                                const priceMatch = priceText.match(/[\d,]+\.?\d*/);
                                if (priceMatch) {
                                    product.price.current.value = parseFloat(priceMatch[0].replace(',', ''));
                                }

                                // Only add if we have at least a name and URL
                                if (product.name && product.url) {
                                    products.push(product);
                                }
                            }
                        } catch (e) {
                            console.log('Error extracting product:', e);
                        }
                    });

                    console.log(`Successfully extracted ${products.length} products`);
                    return products;
                }
            """)

            logger.info(f"Found {len(product_data)} products on page")

            # If no products found, try the search backup URL
            if len(product_data) == 0 and 'search_backup_url' in locals():
                logger.info(f"No products found on category page, trying search backup: {search_backup_url}")
                await page.goto(search_backup_url, wait_until='networkidle', timeout=30000)
                await page.wait_for_timeout(5000)
                await self.scroll_and_wait(page)
                await page.wait_for_timeout(5000)

                product_data = await page.evaluate("""
                    () => {
                        const products = [];
                        const productElements = document.querySelectorAll('a[href*="/prd/"]');

                        productElements.forEach((element, index) => {
                            try {
                                const href = element.href;
                                const match = href.match(/\/prd\/(\d+)/);
                                if (match && !products.some(p => p.id === match[1])) {
                                    const img = element.querySelector('img');
                                    const nameElement = element.querySelector('h3, h4, .title') || element;

                                    products.push({
                                        id: match[1],
                                        name: nameElement?.textContent?.trim() || `Product ${match[1]}`,
                                        url: href,
                                        imageUrl: img?.src || '',
                                        price: {
                                            current: {
                                                value: null,
                                                text: ''
                                            }
                                        },
                                        brandName: 'ASOS',
                                        colour: '',
                                        colourWayId: null,
                                        hasVariantColours: false,
                                        productCode: null,
                                        productType: 'Product'
                                    });
                                }
                            } catch (e) {
                                console.log('Error extracting product from search:', e);
                            }
                        });

                        return products.slice(0, 20); // Limit to 20 products
                    }
                """)

                logger.info(f"Search backup found {len(product_data)} products")

            # Process products with embeddings
            for product in product_data[:20]:  # Limit to first 20 products per category for testing
                # Download image and create embedding
                image_url = product.get('imageUrl')
                embedding = None

                if image_url and image_url.startswith('http'):
                    embedding = await self.download_and_process_image(image_url)

                # Prepare product for database
                db_product = self.prepare_product_for_db(product, embedding, category_info)
                products.append(db_product)

        except Exception as e:
            logger.error(f"Error scraping category {category_url}: {e}")
        finally:
            await page.close()

        return products

    async def load_more_products(self, page, max_clicks=10):
        """Click the ASOS Load More button to load additional products"""
        clicks = 0

        while clicks < max_clicks:
            try:
                # Look for the specific ASOS load more button from user-provided HTML
                load_button = await page.query_selector('a.loadButton_wWQ3F[data-auto-id="loadMoreProducts"]')

                if not load_button:
                    # Try alternative selectors in case they change
                    load_button = await page.query_selector('[data-auto-id="loadMoreProducts"]')
                    if not load_button:
                        load_button = await page.query_selector('.loadButton_wWQ3F')
                        if not load_button:
                            load_button = await page.query_selector('a[href*="page="][class*="loadButton"]')

                if load_button:
                    # Check if button is visible and enabled
                    is_visible = await load_button.is_visible()
                    if not is_visible:
                        logger.info("Load more button not visible, stopping")
                        break

                    logger.info(f"Clicking ASOS load more button (page {clicks + 2})")

                    # Scroll to button to ensure it's in view
                    await load_button.scroll_into_view_if_needed()
                    await page.wait_for_timeout(1000)

                    # Click the button
                    await load_button.click()

                    # Wait for new products to load (ASOS pagination takes time)
                    await page.wait_for_timeout(8000)

                    # Scroll down to ensure new content is visible
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                    await page.wait_for_timeout(2000)

                    clicks += 1

                    # Check if we've reached the end (button might disappear or change)
                    still_exists = await page.query_selector('a.loadButton_wWQ3F[data-auto-id="loadMoreProducts"]')
                    if not still_exists:
                        logger.info("Load more button disappeared, reached end of products")
                        break

                    # Also check if the button's href changed to indicate end
                    href = await load_button.get_attribute('href')
                    if href and 'page=1' in href:  # Sometimes buttons cycle back
                        logger.info("Load more button reset to page 1, reached end")
                        break

                else:
                    logger.info("No ASOS load more button found")
                    break

            except Exception as e:
                logger.warning(f"Error clicking load more button: {e}")
                break

        logger.info(f"Successfully clicked load more button {clicks} times, loaded {clicks + 1} pages")

    async def scroll_and_wait(self, page):
        """Load products using ASOS pagination system"""
        try:
            # Use the specific ASOS load more functionality
            await self.load_more_products(page, max_clicks=10)  # Load up to 10 additional pages

            # Final scroll to ensure all content is accessible
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            await page.wait_for_timeout(2000)

        except Exception as e:
            logger.warning(f"Error during product loading: {e}")

    async def download_and_process_image(self, image_url: str) -> Optional[np.ndarray]:
        """Download image and create embedding"""
        try:
            # Skip if URL is empty or invalid
            if not image_url or not image_url.startswith('http'):
                return None

            # Download image with timeout
            response = requests.get(image_url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
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

        # Default to WOMAN if uncertain
        return 'WOMAN'

    def process_price(self, price_data: Dict) -> tuple:
        """Extract price and currency from price data"""
        current_price = price_data.get('current', {})
        value = current_price.get('value')
        currency = price_data.get('currency', 'GBP')

        # Convert GBP to USD (approximate)
        if currency == 'GBP' and value:
            value = round(value * 1.27, 2)
            currency = 'USD'

        return value, currency

    def create_metadata(self, product_data: Dict, category_info: Optional[Dict] = None) -> str:
        """Create metadata JSON string from product data"""
        metadata = {
            'product_code': product_data.get('productCode'),
            'colour': product_data.get('colour'),
            'colour_way_id': product_data.get('colourWayId'),
            'brand_name': product_data.get('brandName'),
            'has_variant_colours': product_data.get('hasVariantColours'),
            'product_type': product_data.get('productType'),
            'additional_image_urls': [],
            'is_selling_fast': False,
            'is_promotion': False,
            'facets': [],
            'category_info': category_info,
            'scraped_via': 'browser_automation'
        }
        return json.dumps(metadata)

    def prepare_product_for_db(self, product_data: Dict, embedding: Optional[np.ndarray], category_info: Optional[Dict] = None) -> Dict:
        """Prepare product data for database insertion"""
        price_value, currency = self.process_price(product_data.get('price', {}))

        # Create full URLs
        base_url = "https://www.asos.com"
        product_url = product_data.get('url')
        if product_url and not product_url.startswith('http'):
            product_url = f"{base_url}{product_url}"

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

    async def scrape_all_categories(self, max_categories: Optional[int] = None) -> List[Dict]:
        """Scrape products from multiple categories using browser automation"""
        all_products = []

        # Use first N categories for testing if specified
        categories_to_scrape = self.category_ids[:max_categories] if max_categories else self.category_ids[:10]  # Default to 10 for testing

        logger.info(f"Starting to scrape {len(categories_to_scrape)} categories using browser automation")

        for i, category_id in enumerate(categories_to_scrape):
            # Find category info
            category_info = None
            for cat in self.categories:
                if cat['category_id'] == category_id:
                    category_info = cat
                    break

            category_name = category_info.get('title', 'Unknown') if category_info else 'Unknown'
            category_url = category_info.get('web_url') if category_info else None

            # If no category URL, try a search-based approach
            if not category_url:
                # Try searching for the category name
                search_term = category_name.lower().replace(' ', '+').replace('&', 'and')
                category_url = f'https://www.asos.com/search/?q={search_term}'
                logger.info(f"No category URL found, using search: {category_url}")

            # If category URL exists but might be problematic, also try search as backup
            search_backup_url = f'https://www.asos.com/search/?q={category_name.lower().replace(" ", "+").replace("&", "and")}'

            logger.info(f"[{i+1}/{len(categories_to_scrape)}] Scraping category {category_id}: {category_name}")

            try:
                category_products = await self.scrape_category_page(category_url, category_info)
                all_products.extend(category_products)

                logger.info(f"Category {category_id} completed: {len(category_products)} products")

                # Save progress every 5 categories
                if (i + 1) % 5 == 0:
                    await self.save_progress_checkpoint(all_products, i + 1)

                # Longer delay between categories to avoid being blocked
                await asyncio.sleep(10)

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
                'sample_products': products[:5] if products else []
            }

            with open(f'checkpoint_browser_{categories_completed}_categories.json', 'w', encoding='utf-8') as f:
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
                        with open('failed_products_browser.json', 'a', encoding='utf-8') as f:
                            json.dump(product, f, ensure_ascii=False)
                            f.write('\n')
                    except Exception as e3:
                        logger.error(f"Could not save failed product to file: {e3}")

                    continue

        return success_count

    async def run_scraper(self, max_categories: Optional[int] = None):
        """Main scraper execution"""
        logger.info("Starting ASOS Browser-Based Scraper")

        try:
            # Scrape products from categories using browser automation
            logger.info("Starting browser-based category scraping...")
            products = await self.scrape_all_categories(max_categories)

            if not products:
                logger.warning("No products found! ASOS may be blocking browser automation as well.")
                logger.info("Try using the sample data file or check network connectivity.")
                return

            logger.info(f"Successfully processed {len(products)} products from {len(self.category_ids[:max_categories] if max_categories else self.category_ids[:10])} categories")

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
                logger.info(f"Failed products saved to 'failed_products_browser.json' for later retry")

            # Save successful products to JSON file as backup
            try:
                with open('successful_products_browser.json', 'w', encoding='utf-8') as f:
                    json.dump(products, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(products)} products to 'successful_products_browser.json'")
            except Exception as e:
                logger.error(f"Could not save products to JSON file: {e}")

        except Exception as e:
            logger.error(f"Scraper failed: {e}")
            raise
        finally:
            # Cleanup
            await self.close_browser()
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

    scraper = ASOSBrowserScraper()
    await scraper.run_scraper(max_categories)

if __name__ == "__main__":
    asyncio.run(main())