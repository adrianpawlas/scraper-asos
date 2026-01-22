#!/usr/bin/env python3
"""
Test script for ASOS scraper
"""

import asyncio
import json
import aiohttp
from fake_useragent import UserAgent

async def test_api_access():
    """Test if we can access ASOS API"""
    ua = UserAgent()
    headers = {
        'User-Agent': ua.random,
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Test the search API endpoint
    params = {
        'channel': 'mobile-web',
        'country': 'US',
        'currency': 'USD',
        'keyStoreDataversion': '9821',
        'lang': 'en-US',
        'limit': 10,  # Small limit for testing
        'offset': 0,
        'store': 'US',
        'rowlength': 2
    }

    url = "https://www.asos.com/api/product/search/v2/categories"
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    full_url = f"{url}?{query_string}"

    print(f"Testing API access: {full_url}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(full_url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"[SUCCESS] API access successful! Status: {response.status}")

                    if 'products' in data:
                        products = data['products']
                        print(f"Found {len(products)} products")

                        if products:
                            # Show first product as example
                            product = products[0]
                            print(f"\nExample product:")
                            print(f"ID: {product['id']}")
                            print(f"Name: {product['name']}")
                            print(f"Price: {product.get('price', {}).get('current', {}).get('text', 'N/A')}")
                            print(f"Image URL: {product.get('imageUrl', 'N/A')}")
                            print(f"Product URL: {product.get('url', 'N/A')}")

                    return True
                else:
                    print(f"[FAIL] API access failed! Status: {response.status}")
                    print(f"Response: {await response.text()}")
                    return False

        except Exception as e:
            print(f"[ERROR] API access error: {e}")
            return False

async def test_embedding_generation():
    """Test image embedding generation"""
    try:
        import torch
        from transformers import AutoProcessor, AutoModel
        from PIL import Image
        import requests
        from io import BytesIO

        print("\nTesting embedding generation...")

        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
        model = AutoModel.from_pretrained("google/siglip-base-patch16-384")
        model.to(device)
        model.eval()

        print(f"Using device: {device}")
        print("[SUCCESS] Model loaded successfully")

        # Test with a sample image
        test_image_url = "https://images.asos-media.com/products/asos-design-oversized-t-shirt-with-old-fashioned-graphic-in-white/209772779-1-white"

        try:
            response = requests.get(test_image_url, timeout=10)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert('RGB')

            # Generate embedding
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.pooler_output.cpu().numpy().flatten()

            print(f"[SUCCESS] Embedding generated successfully! Shape: {embedding.shape}")
            print(f"Embedding sample: {embedding[:5]}")

            return True

        except Exception as e:
            print(f"[ERROR] Image processing error: {e}")
            return False

    except Exception as e:
        print(f"[ERROR] Embedding model error: {e}")
        return False

async def test_supabase_connection():
    """Test Supabase connection"""
    try:
        from supabase import create_client

        print("\nTesting Supabase connection...")

        supabase_url = "https://yqawmzggcgpeyaaynrjk.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIvoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4"

        supabase = create_client(supabase_url, supabase_key)

        # Try to select from products table
        result = supabase.table('products').select('id').limit(1).execute()

        print("[SUCCESS] Supabase connection successful!")
        print(f"Table exists and is accessible")

        return True

    except Exception as e:
        print(f"[ERROR] Supabase connection error: {e}")
        return False

async def main():
    """Run all tests"""
    print("Testing ASOS Scraper Components")
    print("=" * 50)

    # Test API access
    api_ok = await test_api_access()

    # Test embedding generation
    embedding_ok = await test_embedding_generation()

    # Test Supabase connection
    supabase_ok = await test_supabase_connection()

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"API Access: {'PASS' if api_ok else 'FAIL'}")
    print(f"Embedding Generation: {'PASS' if embedding_ok else 'FAIL'}")
    print(f"Supabase Connection: {'PASS' if supabase_ok else 'FAIL'}")

    if all([api_ok, embedding_ok, supabase_ok]):
        print("\nAll tests passed! Ready to run the scraper.")
        return True
    else:
        print("\nSome tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
