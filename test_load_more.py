#!/usr/bin/env python3
"""
Test script to validate ASOS Load More button functionality
"""

import asyncio
from playwright.async_api import async_playwright
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_load_more_button():
    """Test if we can find and click the ASOS load more button"""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Show browser for testing
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )

        page = await context.new_page()

        try:
            # Test with a category that likely has multiple pages
            test_url = "https://www.asos.com/men/sale/cat/?cid=8409"  # Men's Sale category
            logger.info(f"Testing load more button on: {test_url}")

            await page.goto(test_url, wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(5000)

            # Check initial product count
            initial_count = await page.evaluate("""
                () => document.querySelectorAll('a[href*="/prd/"]').length
            """)
            logger.info(f"Initial product links found: {initial_count}")

            # Look for the load more button
            load_button = await page.query_selector('a.loadButton_wWQ3F[data-auto-id="loadMoreProducts"]')

            if load_button:
                logger.info("✅ Found ASOS load more button!")

                # Get button details
                href = await load_button.get_attribute('href')
                text = await load_button.inner_text()
                is_visible = await load_button.is_visible()

                logger.info(f"Button href: {href}")
                logger.info(f"Button text: {text}")
                logger.info(f"Button visible: {is_visible}")

                # Click the button
                logger.info("Clicking load more button...")
                await load_button.click()

                # Wait for new content
                await page.wait_for_timeout(8000)

                # Check product count after clicking
                after_count = await page.evaluate("""
                    () => document.querySelectorAll('a[href*="/prd/"]').length
                """)
                logger.info(f"Product links after clicking: {after_count}")

                if after_count > initial_count:
                    logger.info("✅ SUCCESS: Load more button worked! More products loaded.")
                else:
                    logger.info("❌ Load more button didn't load more products")

                # Check if button still exists for next page
                still_exists = await page.query_selector('a.loadButton_wWQ3F[data-auto-id="loadMoreProducts"]')
                if still_exists:
                    next_href = await still_exists.get_attribute('href')
                    logger.info(f"Next page button available: {next_href}")
                else:
                    logger.info("Load more button disappeared - reached end")

            else:
                logger.info("❌ ASOS load more button not found")

                # Check what buttons are available
                all_buttons = await page.evaluate("""
                    () => {
                        const buttons = document.querySelectorAll('a, button');
                        const buttonInfo = [];
                        buttons.forEach(btn => {
                            const className = btn.className;
                            const dataId = btn.getAttribute('data-auto-id');
                            const href = btn.getAttribute('href');
                            if (className.includes('load') || dataId?.includes('load') || href?.includes('page=')) {
                                buttonInfo.push({
                                    tag: btn.tagName,
                                    class: className,
                                    dataId: dataId,
                                    href: href,
                                    text: btn.textContent.trim()
                                });
                            }
                        });
                        return buttonInfo.slice(0, 10); // First 10 matches
                    }
                """)

                logger.info("Available load-related buttons:")
                for btn in all_buttons:
                    logger.info(f"  {btn}")

        except Exception as e:
            logger.error(f"Test failed: {e}")

        finally:
            await page.wait_for_timeout(5000)  # Keep browser open for 5 seconds to see results
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test_load_more_button())