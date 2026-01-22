#!/usr/bin/env python3
"""
Extract all ASOS category IDs and information from navigation data
"""

import json
import re
from typing import List, Dict, Set
from collections import defaultdict

def extract_categories_from_navigation():
    """Extract all category information from 2.txt navigation data"""

    with open('2.txt', 'r', encoding='utf-8') as f:
        nav_data = json.load(f)

    categories = []
    processed_urls = set()

    def process_nav_item(item, parent_path=""):
        """Recursively process navigation items"""
        if not isinstance(item, dict):
            return

        # Extract category info
        content = item.get('content', {})
        title = content.get('title')
        link = item.get('link', {})

        if link and link.get('webUrl'):
            web_url = link['webUrl']

            # Extract category ID from URL
            cid_match = re.search(r'cid=(\d+)', web_url)
            if cid_match:
                cid = cid_match.group(1)

                # Avoid duplicates
                if web_url not in processed_urls:
                    processed_urls.add(web_url)

                    category_info = {
                        'id': item.get('id'),
                        'category_id': cid,
                        'title': title,
                        'web_url': web_url,
                        'app_url': link.get('appUrl'),
                        'parent_path': parent_path,
                        'type': item.get('type'),
                        'alias': item.get('alias')
                    }
                    categories.append(category_info)

        # Process children
        children = item.get('children', [])
        current_path = f"{parent_path} > {title}" if parent_path else title

        for child in children:
            process_nav_item(child, current_path)

    # Process all top-level navigation items
    for item in nav_data:
        process_nav_item(item)

    return categories

def categorize_by_section(categories):
    """Group categories by main sections (Men, Women, Sale, etc.)"""

    sections = defaultdict(list)

    for cat in categories:
        url = cat['web_url'].lower()

        if '/men/' in url:
            sections['men'].append(cat)
        elif '/women/' in url:
            sections['women'].append(cat)
        elif '/sale/' in url:
            sections['sale'].append(cat)
        elif '/new-in/' in url or 'new' in url.lower():
            sections['new_in'].append(cat)
        elif '/outlet/' in url:
            sections['outlet'].append(cat)
        elif '/marketplace/' in url:
            sections['marketplace'].append(cat)
        else:
            sections['other'].append(cat)

    return dict(sections)

def save_categories_to_file(categories, sections):
    """Save extracted categories to JSON files"""

    # Save all categories
    with open('asos_categories.json', 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)

    # Save categorized sections
    with open('asos_category_sections.json', 'w', encoding='utf-8') as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    # Save just category IDs for easy use
    category_ids = list(set(cat['category_id'] for cat in categories))
    with open('asos_category_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(category_ids)))

    print(f"Extracted {len(categories)} categories from navigation data")
    print(f"Found {len(category_ids)} unique category IDs")

    print("\nCategories by section:")
    for section, cats in sections.items():
        print(f"  {section.title()}: {len(cats)} categories")

def main():
    print("Extracting ASOS categories from navigation data...")

    categories = extract_categories_from_navigation()
    sections = categorize_by_section(categories)

    save_categories_to_file(categories, sections)

    print("\nFiles created:")
    print("  - asos_categories.json (all category details)")
    print("  - asos_category_sections.json (grouped by section)")
    print("  - asos_category_ids.txt (just the IDs for API calls)")

if __name__ == "__main__":
    main()

