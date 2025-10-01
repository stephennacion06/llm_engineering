#!/usr/bin/env python3
"""
Confluence Page Summarizer
A standalone script to extract and summarize Confluence pages using Google Gemini API

Usage:
    python confluence_summarizer.py <confluence_url>
    python confluence_summarizer.py --interactive
    python confluence_summarizer.py --help

Requirements:
    - .env file with GOOGLE_API_KEY, CONFLUENCE_TOKEN, CONFLUENCE_URL
    - pip install requests beautifulsoup4 python-dotenv google-generativeai urllib3
"""

import os
import sys
import argparse
import requests
import urllib3
from urllib.parse import urlparse, unquote
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Disable SSL warnings for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ConfluencePageSummarizer:
    """Main class for Confluence page summarization"""
    
    def __init__(self):
        """Initialize the summarizer with environment variables"""
        self.load_environment()
        self.setup_gemini_client()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }
    
    def load_environment(self):
        """Load environment variables from .env file"""
        load_dotenv(override=True)
        
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.confluence_token = os.getenv('CONFLUENCE_TOKEN')
        self.confluence_url = os.getenv('CONFLUENCE_URL')
        
        # Validate required environment variables
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Debug: Check what the key actually starts with
        print(f"🔍 API key starts with: {self.google_api_key[:10]}...")
        
        # More flexible validation - Google keys can have different formats
        if len(self.google_api_key) < 30:
            raise ValueError("Google API key seems too short")
        
        print("✅ Environment variables loaded successfully")
    
    def setup_gemini_client(self):
        """Setup Gemini API client"""
        try:
            from google import genai
            self.client = genai.Client(api_key=self.google_api_key)
            print("✅ Gemini client initialized successfully")
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    def mask_sensitive_data(self, text):
        """Mask sensitive data before processing"""
        if not text:
            return text
        
        # Replace sensitive patterns - Generic company masking
        patterns = [
            # Email addresses
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]'),
            # API keys
            (r'AIza[0-9A-Za-z_-]{35}', '[API_KEY_MASKED]'),
            (r'sk-[A-Za-z0-9_-]+', '[API_KEY_MASKED]'),
            (r'NTYx[A-Za-z0-9_-]+', '[TOKEN_MASKED]'),
            # Generic company URLs and domains
            (r'https://confluence\.[a-zA-Z0-9.-]+\.(com|corp|net|org)', 'https://[COMPANY_CONFLUENCE]'),
            (r'https://[a-zA-Z0-9.-]+\.corp\b', 'https://[COMPANY_DOMAIN]'),
            # Internal URLs and IPs
            (r'https?://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', 'http://[INTERNAL_IP]'),
            (r'https?://[a-zA-Z0-9.-]*internal[a-zA-Z0-9.-]*', 'https://[INTERNAL_DOMAIN]'),
            # Long numbers that might be IDs
            (r'\b\d{6,}\b', '[ID_MASKED]'),
            # Phone numbers
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]'),
            # Generic company names in URLs and text
            (r'\b[A-Z][a-z]+\s+(Corp|Corporation|Inc|Ltd|LLC)\b', '[COMPANY_NAME]'),
        ]
        
        masked_text = text
        for pattern, replacement in patterns:
            masked_text = re.sub(pattern, replacement, masked_text, flags=re.IGNORECASE)
        
        return masked_text
    
    def extract_page_info_from_url(self, url):
        """Extract space key and page title from Confluence URL"""
        # URL formats:
        # /display/SPACEKEY/Page+Title
        # /pages/viewpage.action?pageId=123456
        
        page_info = {}
        
        match = re.search(r'/display/([^/]+)/(.+)', url)
        if match:
            page_info['space_key'] = match.group(1)
            page_info['page_title'] = unquote(match.group(2)).replace('+', ' ')
            return page_info
        
        match = re.search(r'pageId=(\d+)', url)
        if match:
            page_info['page_id'] = match.group(1)
            return page_info
        
        return page_info
    
    def fetch_via_api(self, url, page_info):
        """Fetch content using Confluence REST API"""
        if not self.confluence_token or not self.confluence_url:
            return None
        
        try:
            session = requests.Session()
            session.headers.update({
                'Authorization': f'Bearer {self.confluence_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
            if 'page_id' in page_info:
                # Direct page ID lookup
                api_url = f"{self.confluence_url}/rest/api/content/{page_info['page_id']}"
                params = {'expand': 'body.storage,space,version,title'}
                response = session.get(api_url, params=params, verify=False)
            
            elif 'space_key' in page_info and 'page_title' in page_info:
                # Search by title in space
                api_url = f"{self.confluence_url}/rest/api/content/search"
                cql_query = f"space='{page_info['space_key']}' AND title~'{page_info['page_title']}'"
                params = {
                    'cql': cql_query,
                    'expand': 'body.storage,space,version,title',
                    'limit': 1
                }
                response = session.get(api_url, params=params, verify=False)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        content = data['results'][0]
                    else:
                        return None
                else:
                    return None
            else:
                return None
            
            if response.status_code == 200:
                if 'results' not in locals():
                    content = response.json()
                
                title = content.get('title', 'No title')
                
                # Extract text from storage format
                if content.get('body', {}).get('storage', {}).get('value'):
                    storage_content = content['body']['storage']['value']
                    # Parse HTML and extract clean text
                    soup = BeautifulSoup(storage_content, 'html.parser')
                    # Remove unwanted elements
                    for tag in soup(['script', 'style', 'meta', 'link']):
                        tag.decompose()
                    text = soup.get_text(separator='\n', strip=True)
                    
                    return {'title': title, 'text': text, 'method': 'API'}
                else:
                    return None
            else:
                print(f"API Error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"API fetch failed: {e}")
            return None
    
    def fetch_via_web_scraping(self, url):
        """Fallback to web scraping"""
        try:
            session = requests.Session()
            
            # Add token to headers if available
            if self.confluence_token:
                session.headers.update({
                    'Authorization': f'Bearer {self.confluence_token}',
                    'X-Atlassian-Token': 'no-check'
                })
            
            session.headers.update(self.headers)
            
            response = session.get(url, verify=False, timeout=30)
            
            if response.status_code != 200:
                return {
                    'title': f"HTTP Error {response.status_code}",
                    'text': "Could not fetch page content",
                    'method': 'Web Scraping (Failed)'
                }
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No title found"
            
            # Look for Confluence-specific content areas
            content_areas = [
                {'id': 'main-content'},
                {'class': 'wiki-content'},
                {'class': 'page-content'},
                {'id': 'content'},
                {'class': 'main-content'}
            ]
            
            main_content = None
            for selector in content_areas:
                main_content = soup.find('div', selector)
                if main_content:
                    break
            
            # If no specific content area found, use body
            if not main_content:
                main_content = soup.body
            
            if main_content:
                # Remove navigation, sidebar, and other irrelevant elements
                for irrelevant in main_content(['script', 'style', 'img', 'input', 
                                              'nav', 'header', 'footer', 'sidebar',
                                              'breadcrumbs', 'comment']):
                    irrelevant.decompose()
                
                # Remove elements with navigation-related classes
                nav_classes = ['nav', 'navigation', 'menu', 'sidebar', 'breadcrumb', 
                              'header', 'footer', 'comment', 'metadata']
                for nav_class in nav_classes:
                    for element in main_content.find_all(class_=re.compile(nav_class, re.I)):
                        element.decompose()
                
                text = main_content.get_text(separator="\n", strip=True)
                return {'title': title, 'text': text, 'method': 'Web Scraping'}
            else:
                return {'title': title, 'text': "No content found", 'method': 'Web Scraping'}
                
        except Exception as e:
            print(f"Web scraping failed: {e}")
            return {
                'title': "Fetch Failed",
                'text': f"Error fetching content: {e}",
                'method': 'Web Scraping (Failed)'
            }
    
    def extract_content(self, url, use_api=True, mask_data=True):
        """Extract content from Confluence page"""
        print(f"🔍 Fetching content from: {url}")
        
        # Parse URL for page information
        page_info = self.extract_page_info_from_url(url)
        
        # Try API first if enabled
        content = None
        if use_api:
            content = self.fetch_via_api(url, page_info)
            if content:
                print(f"✅ Content fetched via API")
            else:
                print("⚠️ API failed, falling back to web scraping...")
        
        # Fallback to web scraping
        if not content:
            content = self.fetch_via_web_scraping(url)
            print(f"✅ Content fetched via {content['method']}")
        
        # Mask sensitive data if enabled
        if mask_data:
            content['text'] = self.mask_sensitive_data(content['text'])
            content['title'] = self.mask_sensitive_data(content['title'])
        
        return content
    
    def create_summary_prompt(self, content):
        """Create optimized prompt for Confluence content summarization"""
        system_prompt = """You are an AI assistant specialized in analyzing and summarizing technical documentation and internal knowledge base content. 
        
        Your task is to:
        1. Identify the main purpose and scope of the document
        2. Extract key technical information, procedures, or findings
        3. Highlight important conclusions or recommendations
        4. Present the summary in clear, structured markdown
        5. Ignore navigation elements, metadata, or administrative content
        
        Focus on the substantive content that would be valuable for someone trying to understand the topic."""
        
        user_prompt = f"""You are analyzing a Confluence page titled: '{content['title']}'

This appears to be internal documentation or knowledge base content. 
Please provide a comprehensive summary that captures:
- Main purpose/objective of the document
- Key technical details or findings
- Important procedures or steps mentioned
- Any conclusions or recommendations

Content to analyze:

{content['text']}"""
        
        return system_prompt, user_prompt
    
    def summarize_with_gemini(self, content):
        """Generate summary using Gemini API"""
        print(f"📄 Page title: {content['title']}")
        print(f"📊 Content length: {len(content['text'])} characters")
        
        if len(content['text']) < 100:
            return "⚠️ **Warning**: Very little content was extracted."
        
        try:
            system_prompt, user_prompt = self.create_summary_prompt(content)
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            
            print("🤖 Generating summary with Gemini...")
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=full_prompt
            )
            
            return response.text
            
        except Exception as e:
            print(f"❌ Full error details: {e}")
            return f"❌ Error calling Gemini API: {e}"
    
    def summarize_page(self, url, use_api=True, mask_data=True, output_file=None):
        """Main method to summarize a Confluence page"""
        try:
            # Extract content
            content = self.extract_content(url, use_api=use_api, mask_data=mask_data)
            
            # Generate summary
            summary = self.summarize_with_gemini(content)
            
            # Output results
            result = f"""
# Confluence Page Summary

**Original URL:** {url}
**Page Title:** {content['title']}
**Extraction Method:** {content['method']}
**Content Length:** {len(content['text'])} characters

---

## Summary

{summary}

---

*Generated by Confluence Page Summarizer*
"""
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"💾 Summary saved to: {output_file}")
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error summarizing page: {e}"
            print(error_msg)
            return error_msg

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Confluence Page Summarizer using Google Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python confluence_summarizer.py https://confluence.company.com/display/SPACE/Page+Title
    python confluence_summarizer.py --interactive
    python confluence_summarizer.py -u https://confluence.company.com/page --output summary.md
    python confluence_summarizer.py -u https://confluence.company.com/page --no-api --no-mask
        """
    )
    
    parser.add_argument('url', nargs='?', help='Confluence page URL to summarize')
    parser.add_argument('-u', '--url', dest='url_flag', help='Confluence page URL to summarize')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('-o', '--output', help='Output file path for summary')
    parser.add_argument('--no-api', action='store_true', help='Skip API and use only web scraping')
    parser.add_argument('--no-mask', action='store_true', help='Disable sensitive data masking')
    parser.add_argument('--version', action='version', version='Confluence Summarizer 1.0')
    
    args = parser.parse_args()
    
    # Determine URL
    url = args.url or args.url_flag
    
    try:
        # Initialize summarizer
        print("🚀 Initializing Confluence Page Summarizer...")
        summarizer = ConfluencePageSummarizer()
        
        # Interactive mode
        if args.interactive or not url:
            print("\n" + "="*50)
            print("Interactive Confluence Page Summarizer")
            print("="*50)
            
            while True:
                if not url:
                    url = input("\nEnter Confluence page URL (or 'quit' to exit): ").strip()
                
                if url.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not url.startswith('http'):
                    print("❌ Please enter a valid URL starting with http:// or https://")
                    url = None
                    continue
                
                # Generate summary
                print(f"\n📝 Processing: {url}")
                summary = summarizer.summarize_page(
                    url, 
                    use_api=not args.no_api,
                    mask_data=not args.no_mask,
                    output_file=args.output
                )
                
                print("\n" + "="*50)
                print("SUMMARY")
                print("="*50)
                print(summary)
                
                # Ask for another URL
                url = None
        
        else:
            # Single URL mode
            print(f"\n📝 Processing: {url}")
            summary = summarizer.summarize_page(
                url, 
                use_api=not args.no_api,
                mask_data=not args.no_mask,
                output_file=args.output
            )
            
            print("\n" + "="*50)
            print("SUMMARY")
            print("="*50)
            print(summary)
    
    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your .env file has GOOGLE_API_KEY, CONFLUENCE_TOKEN, CONFLUENCE_URL")
        print("2. Ensure all required packages are installed: pip install -r requirements.txt")
        print("3. Verify your network connection and VPN if required")
        sys.exit(1)

if __name__ == "__main__":
    main()