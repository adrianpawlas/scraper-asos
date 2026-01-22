#!/usr/bin/env python3
"""
Test different ASOS API endpoints to find the correct way to query categories
"""

import requests
import json

def test_api_endpoints():
    """Test different API endpoints and parameters"""

    categories_to_try = ['new-in-today', 'men', 'women', 'sale']

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    for category in categories_to_try:
        print(f'\nTrying category: {category}')

        # Try different API endpoints
        endpoints = [
            f'https://www.asos.com/api/product/search/v2/categories?channel=desktop&country=US&currency=USD&keyStoreDataversion=9821&lang=en-US&limit=10&offset=0&store=US&q={category}',
            f'https://www.asos.com/api/product/search/v2/list?channel=desktop&country=US&currency=USD&keyStoreDataversion=9821&lang=en-US&limit=10&offset=0&store=US&category={category}',
            f'https://www.asos.com/api/product/list/v2/{category}?channel=desktop&country=US&currency=USD&keyStoreDataversion=9821&lang=en-US&limit=10&offset=0&store=US',
            f'https://www.asos.com/api/product/search/v2/categories?channel=desktop&country=US&currency=USD&keyStoreDataversion=9821&lang=en-US&limit=10&offset=0&store=US&sort=newest'
        ]

        for i, endpoint in enumerate(endpoints):
            try:
                print(f'  Endpoint {i+1}: {endpoint[:80]}...')
                response = requests.get(endpoint, headers=headers, timeout=10)
                print(f'  Status: {response.status}')

                if response.status_code == 200:
                    try:
                        data = response.json()
                        products = data.get('products', [])
                        print(f'  Products found: {len(products)}')

                        if products:
                            sample_name = products[0].get('name', '')[:50]
                            print(f'  Sample product: {sample_name}...')
                            print('  ✅ SUCCESS! This endpoint works.')
                            return endpoint, data  # Return successful endpoint and data

                    except json.JSONDecodeError:
                        print(f'  Not JSON response: {response.text[:100]}')

                elif response.status_code == 404:
                    print('  404 Not Found')
                elif response.status_code == 403:
                    print('  403 Forbidden')
                else:
                    print(f'  Other status: {response.status_code}')

            except Exception as e:
                print(f'  Error: {e}')

    return None, None

def test_with_category_ids():
    """Test using category IDs directly"""

    print('\n\nTesting with category IDs...')

    # Try some category IDs from our extracted data
    category_ids = ['2097', '26954', '19286']  # From the extracted data

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    for cid in category_ids:
        print(f'\nTesting category ID: {cid}')

        # Try different parameter names
        param_names = ['cid', 'categoryId', 'category_id', 'id']

        for param in param_names:
            endpoint = f'https://www.asos.com/api/product/search/v2/categories?channel=desktop&country=US&currency=USD&keyStoreDataversion=9821&lang=en-US&limit=10&offset=0&store=US&{param}={cid}'
            try:
                print(f'  Param {param}: ', end='')
                response = requests.get(endpoint, headers=headers, timeout=10)
                print(f'Status {response.status}')

                if response.status_code == 200:
                    try:
                        data = response.json()
                        products = data.get('products', [])
                        print(f'  Products: {len(products)}')

                        if products:
                            print('  ✅ SUCCESS with category ID!')
                            return endpoint, data

                    except json.JSONDecodeError:
                        pass

            except Exception as e:
                print(f'Error: {e}')

    return None, None

if __name__ == "__main__":
    print("Testing ASOS API endpoints...")

    # Test general category queries
    endpoint1, data1 = test_api_endpoints()

    if endpoint1:
        print(f'\n🎉 Found working endpoint: {endpoint1}')
        with open('working_endpoint_sample.json', 'w') as f:
            json.dump(data1, f, indent=2)
    else:
        print('\n❌ No working general category endpoint found')

    # Test category ID queries
    endpoint2, data2 = test_with_category_ids()

    if endpoint2:
        print(f'\n🎉 Found working category ID endpoint: {endpoint2}')
        with open('working_category_id_sample.json', 'w') as f:
            json.dump(data2, f, indent=2)
    else:
        print('\n❌ No working category ID endpoint found')

    print('\n📝 Next steps:')
    print('1. Use the working endpoints in the scraper')
    print('2. Update the multi-category scraper with correct API calls')
    print('3. Test with more categories')

