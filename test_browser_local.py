#!/usr/bin/env python3
"""
Test browser scraper locally before pushing to GitHub
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asos_scraper_browser import ASOSBrowserScraper
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_local():
    """Test browser scraper locally with just 1 category"""
    print("=" * 60)
    print("Testing ASOS Browser Scraper Locally")
    print("=" * 60)
    print()

    scraper = ASOSBrowserScraper()

    try:
        # Test with just 1 category
        print("Testing with 1 category...")
        products = await scraper.scrape_all_categories(max_categories=1)

        if products:
            print(f"\n✅ SUCCESS! Found {len(products)} products")
            print(f"\nSample product:")
            print(f"  Title: {products[0]['title'][:60]}...")
            print(f"  Gender: {products[0]['gender']}")
            print(f"  Price: {products[0]['price']} {products[0]['currency']}")
            print(f"  Has embedding: {products[0]['embedding'] is not None}")
            return True
        else:
            print("\n❌ FAILED! No products found")
            print("Browser automation is being blocked by ASOS")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await scraper.close_browser()

if __name__ == "__main__":
    success = asyncio.run(test_local())
    sys.exit(0 if success else 1)