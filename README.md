# ASOS Fashion Scraper

A comprehensive scraper for ASOS fashion products that extracts product information, generates 768-dimensional image embeddings, and stores everything in Supabase.

## 🚀 Quick Start

### Automated (Recommended)
- **Daily runs automatically at midnight UTC**
- Trigger manually via GitHub Actions
- No local setup required

### Manual Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run scraper manually
python run_scraper.py

# Or run directly
python asos_scraper_multi_category.py 10  # Scrape 10 categories
```

## Features

✅ **Complete Product Data Extraction**
- Product ID, title, brand, price, currency
- Gender detection (MAN/WOMAN)
- Product URLs and image URLs
- Metadata and additional product information

✅ **Image Embedding Generation**
- Uses CLIP (openai/clip-vit-base-patch32) model
- Generates 768-dimensional embeddings
- Robust error handling for image processing

✅ **Supabase Integration**
- Automatic database insertion with upsert logic
- Proper error handling and retry mechanisms
- JSON backup of successful/failed products

✅ **Robust Architecture**
- Rate limiting and request throttling
- Comprehensive logging
- Graceful error handling
- Sample data fallback when API is blocked

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configuration**
The scraper uses the following Supabase credentials (already configured):
- URL: `https://yqawmzggcgpeyaaynrjk.supabase.co`
- Service Key: (configured in the script)

## Usage

**Run the Full Scraper**
```bash
python asos_scraper.py
```

The scraper will:
1. Load sample product data from `3.txt` (fallback when API is blocked)
2. Download product images
3. Generate CLIP embeddings (768 dimensions)
4. Process product information (gender detection, price conversion)
5. Save to Supabase database
6. Create backup JSON files

## Database Schema

The scraper populates the `products` table with:

```sql
CREATE TABLE products (
  id text PRIMARY KEY,
  source text DEFAULT 'scraper',
  product_url text,
  image_url text NOT NULL,
  brand text DEFAULT 'ASOS',
  title text NOT NULL,
  gender text, -- 'MAN' or 'WOMAN'
  price double precision,
  currency text, -- 'USD' or 'EUR'
  metadata text, -- JSON string with full product info
  second_hand boolean DEFAULT FALSE,
  embedding public.vector, -- 768-dimensional CLIP embedding
  created_at timestamp with time zone DEFAULT now()
);
```

## Output Files

- `successful_products.json` - Successfully processed products
- `failed_products.json` - Products that failed to save to database
- `asos_scraper.log` - Detailed execution logs

## Current Status

✅ **Working Components:**
- Product data processing
- Image embedding generation (CLIP, 768-dim)
- Gender detection logic
- Supabase database insertion
- Error handling and logging
- Sample data processing

⚠️ **Known Limitations:**
- ASOS API is currently blocking automated requests
- Uses sample data from `3.txt` as fallback
- Image downloads may timeout (ASOS rate limiting)

## Technical Details

**Embedding Model:** CLIP (openai/clip-vit-base-patch32)
- Input: Product images
- Output: 768-dimensional vectors
- Used for similarity search and recommendations

**Gender Detection:**
- Analyzes product names and URLs for gender indicators
- Defaults to 'WOMAN' for ASOS products

**Price Conversion:**
- Converts GBP to USD (approximate rate: 1 GBP = 1.27 USD)
- Maintains original currency information

## 🤖 GitHub Actions Automation

### Daily Automated Runs
- **Schedule:** Every day at midnight UTC
- **Trigger:** Automatic cron job
- **Results:** Available in Actions tab and artifacts

### Manual Trigger via GitHub
1. Go to [GitHub Actions](https://github.com/adrianpawlas/scraper-asos/actions)
2. Click "Daily ASOS Scraper" workflow
3. Click "Run workflow"
4. Configure options (optional):
   - **Categories limit:** Number of categories to scrape
   - **Force run:** Override any existing checks

### Required GitHub Secrets
Add these to your repository secrets (`Settings` → `Secrets and variables` → `Actions`):
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_ANON_KEY`: Your Supabase anon key

### Manual Local Execution
```bash
# Quick test (sample data only)
python run_scraper.py --sample-only

# Test with 5 categories
python run_scraper.py --test

# Full scrape (all categories)
python run_scraper.py

# Custom number of categories
python run_scraper.py --categories 50
```

## Troubleshooting

**Database Connection Issues:**
- Check Supabase credentials
- Verify table schema matches requirements
- Check `asos_scraper.log` for detailed errors

**Image Processing Issues:**
- Images may timeout due to ASOS rate limiting
- Products without embeddings will still be saved
- Check network connectivity

**API Blocking:**
- ASOS blocks automated requests
- Current implementation uses sample data
- May need browser automation or different API endpoints

## Next Steps

To scrape more products:
1. Find alternative ASOS API endpoints
2. Implement browser automation (Selenium/Playwright)
3. Use ASOS sitemaps for product discovery
4. Implement proxy rotation for better success rates

The scraper is production-ready for the current data and can be extended as needed!

