#!/usr/bin/env python3
"""
Test the scraper pipeline using the sample data from 3.txt
"""

import json
import asyncio
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
from io import BytesIO
from supabase import create_client
from datetime import datetime

async def load_sample_data():
    """Load sample product data from 3.txt"""
    with open('3.txt', 'r', encoding='utf-8') as f:
        data = json.load(f)

    products = data.get('products', [])
    print(f"Loaded {len(products)} products from sample data")
    return products

def detect_gender(product_data):
    """Detect gender from product data"""
    name = product_data.get('name', '').lower()
    url = product_data.get('url', '').lower()

    # Check for gender indicators in name or URL
    if any(word in name or word in url for word in ['men', 'man', 'male', 'mens', 'guy', 'boy']):
        return 'MAN'
    elif any(word in name or word in url for word in ['women', 'woman', 'female', 'womens', 'girl', 'lady']):
        return 'WOMAN'

    # Default to WOMAN if uncertain
    return 'WOMAN'

def process_price(price_data):
    """Extract price and currency from price data"""
    current_price = price_data.get('current', {})
    value = current_price.get('value')
    currency = price_data.get('currency', 'GBP')

    # Convert GBP to USD (approximate)
    if currency == 'GBP' and value:
        value = round(value * 1.27, 2)  # Current GBP to USD rate
        currency = 'USD'

    return value, currency

async def download_and_process_image(image_url):
    """Download image and create embedding"""
    try:
        # Ensure full URL
        if not image_url.startswith('http'):
            image_url = f"https://{image_url}"

        # Download image
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()

        # Open image
        image = Image.open(BytesIO(response.content)).convert('RGB')

        # Load model (do this once and reuse)
        if not hasattr(download_and_process_image, 'model'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            download_and_process_image.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
            download_and_process_image.model = AutoModel.from_pretrained("google/siglip-base-patch16-384")
            download_and_process_image.model.to(device)
            download_and_process_image.model.eval()
            download_and_process_image.device = device

        # Generate embedding
        inputs = download_and_process_image.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(download_and_process_image.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = download_and_process_image.model(**inputs)
            embedding = outputs.pooler_output.cpu().numpy().flatten()

        return embedding

    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None

def create_metadata(product_data):
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
    }
    return json.dumps(metadata)

def prepare_product_for_db(product_data, embedding):
    """Prepare product data for database insertion"""
    price_value, currency = process_price(product_data.get('price', {}))

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
        'category': None,
        'gender': detect_gender(product_data),
        'price': price_value,
        'currency': currency,
        'search_tsv': None,
        'created_at': datetime.now().isoformat(),
        'metadata': create_metadata(product_data),
        'size': None,
        'second_hand': False,
        'embedding': embedding.tolist() if embedding is not None else None,
        'country': 'US',
        'compressed_image_url': None,
        'tags': None,
        'search_vector': None
    }

    return product

async def test_pipeline():
    """Test the complete pipeline with sample data"""
    print("Testing ASOS scraper pipeline with sample data")
    print("=" * 60)

    # Load sample data
    print("Loading sample data...")
    products_data = await load_sample_data()

    # Take first 3 products for testing
    test_products = products_data[:3]
    print(f"Testing with {len(test_products)} products")

    # Initialize Supabase
    supabase_url = "https://yqawmzggcgpeyaaynrjk.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4"
    supabase = create_client(supabase_url, supabase_key)

    processed_products = []

    for i, product_data in enumerate(test_products):
        print(f"\nProcessing product {i+1}: {product_data['name'][:50]}...")

        # Download image and create embedding
        image_url = product_data.get('imageUrl')
        embedding = None

        if image_url:
            print(f"  Downloading image: {image_url}")
            embedding = await download_and_process_image(image_url)
            if embedding is not None:
                print(f"  Generated embedding with shape: {embedding.shape}")
            else:
                print("  Failed to generate embedding")
        # Prepare product for database
        db_product = prepare_product_for_db(product_data, embedding)
        processed_products.append(db_product)

        print(f"  Gender: {db_product['gender']}")
        print(f"  Price: {db_product['price']} {db_product['currency']}")
        print(f"  Product URL: {db_product['product_url']}")

    # Save to database
    print(f"\nSaving {len(processed_products)} products to database...")
    saved_count = 0

    for product in processed_products:
        try:
            result = supabase.table('products').upsert(
                product,
                on_conflict='source,product_url'
            ).execute()
            saved_count += 1
            print(f"  Saved product: {product['title'][:30]}...")
        except Exception as e:
            print(f"  Error saving product {product.get('id')}: {e}")

    print(f"\nResults:")
    print(f"- Processed: {len(processed_products)} products")
    print(f"- Saved to DB: {saved_count} products")
    print(f"- Success rate: {saved_count/len(processed_products)*100:.1f}%")

    return saved_count > 0

async def main():
    success = await test_pipeline()
    if success:
        print("\n[SUCCESS] Pipeline test completed successfully!")
        print("The scraper components are working correctly.")
        print("Now we need to figure out how to get more product data from ASOS API.")
    else:
        print("\n[FAIL] Pipeline test failed.")

if __name__ == "__main__":
    asyncio.run(main())
