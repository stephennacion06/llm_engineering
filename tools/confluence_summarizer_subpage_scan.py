#!/usr/bin/env python3
"""
Confluence Deep Page Analyzer with Full Subpage Support v3.0
A script to extract and summarize ALL Confluence pages including deeply nested subpages

IMPROVEMENTS IN V3.0:
- Better handling of failed page fetches
- Retries for API failures
- Web scraping fallback for child discovery
- More detailed logging
- Handles "No title" pages better

Usage:
    python confluence_summarizer_subpage_scan.py <confluence_url>
    python confluence_summarizer_subpage_scan.py --interactive
    python confluence_summarizer_subpage_scan.py --help

Requirements:
    - .env file with GOOGLE_API_KEY, CONFLUENCE_TOKEN, CONFLUENCE_URL
    - pip install requests beautifulsoup4 python-dotenv google-generativeai urllib3
"""

import os
import sys
import argparse
import requests
import urllib3
from urllib.parse import urlparse, unquote, urljoin, parse_qs
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import List, Dict, Set, Optional
import time

# Disable SSL warnings for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ConfluenceDeepAnalyzer:
    """Main class for deep Confluence page analysis with complete subpage support"""
    
    def __init__(self):
        """Initialize the analyzer with environment variables"""
        self.load_environment()
        self.setup_gemini_client()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }
        self.visited_pages: Set[str] = set()
        self.visited_page_ids: Set[str] = set()
        self.max_depth = 10
        self.total_pages = 0
        self.failed_pages = []
        
    def load_environment(self):
        """Load environment variables from .env file"""
        load_dotenv(override=True)
        
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.confluence_token = os.getenv('CONFLUENCE_TOKEN')
        self.confluence_url = os.getenv('CONFLUENCE_URL')
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        if len(self.google_api_key) < 30:
            raise ValueError("Google API key seems too short")
        
        print("✅ Environment variables loaded successfully")
    
    def setup_gemini_client(self):
        """Setup Gemini API client"""
        try:
            from google import genai
            self.client = genai.Client(api_key=self.google_api_key)
            print("✅ Gemini client initialized successfully")
            print("🔒 Privacy: Google's API terms ensure your data is not used for training")
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    def mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data before processing - PRIVACY CRITICAL"""
        if not text:
            return text
        
        patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
            (r'AIza[0-9A-Za-z_-]{35}', '[API_KEY_REDACTED]'),
            (r'sk-[A-Za-z0-9_-]{20,}', '[API_KEY_REDACTED]'),
            (r'NTYx[A-Za-z0-9_-]+', '[TOKEN_REDACTED]'),
            (r'Bearer [A-Za-z0-9_-]+', 'Bearer [TOKEN_REDACTED]'),
            (r'https://confluence\.[a-zA-Z0-9.-]+\.(com|corp|net|org)', 'https://[COMPANY_CONFLUENCE_REDACTED]'),
            (r'https://[a-zA-Z0-9.-]+\.corp\b', 'https://[COMPANY_DOMAIN_REDACTED]'),
            (r'https?://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', 'http://[INTERNAL_IP_REDACTED]'),
            (r'\b\d{6,}\b', '[ID_REDACTED]'),
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]'),
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),
            (r'\b[A-Z][a-z]+\s+(Corp|Corporation|Inc|Ltd|LLC)\b', '[COMPANY_NAME_REDACTED]'),
            (r'@[A-Za-z0-9_]+', '@[USERNAME_REDACTED]'),
            (r'[C-Z]:\\[^\\s]+', '[FILEPATH_REDACTED]'),
            (r'/home/[a-zA-Z0-9_-]+', '/home/[USER_REDACTED]'),
            (r'/Users/[a-zA-Z0-9_-]+', '/Users/[USER_REDACTED]'),
        ]
        
        masked_text = text
        for pattern, replacement in patterns:
            masked_text = re.sub(pattern, replacement, masked_text, flags=re.IGNORECASE)
        
        return masked_text
    
    def extract_page_id_from_url(self, url: str) -> Optional[str]:
        """Extract page ID from any Confluence URL format"""
        # Try pageId parameter
        match = re.search(r'pageId=(\d+)', url)
        if match:
            return match.group(1)
        
        # Try from API response URL format
        match = re.search(r'/pages/(\d+)/', url)
        if match:
            return match.group(1)
        
        return None
    
    def extract_page_info_from_url(self, url: str) -> Dict:
        """Extract space key and page title from Confluence URL"""
        page_info = {}
        
        # Extract page ID
        page_id = self.extract_page_id_from_url(url)
        if page_id:
            page_info['page_id'] = page_id
        
        # Extract space and title from display URL
        match = re.search(r'/display/([^/]+)/(.+)', url)
        if match:
            page_info['space_key'] = match.group(1)
            page_info['page_title'] = unquote(match.group(2)).replace('+', ' ')
        
        return page_info
    
    def get_all_child_pages_from_api(self, page_id: str, retry_count: int = 3) -> List[Dict]:
        """Get ALL child pages using Confluence REST API with pagination and retries"""
        if not self.confluence_token or not self.confluence_url or not page_id:
            return []
        
        all_children = []
        start = 0
        limit = 100
        
        for attempt in range(retry_count):
            try:
                session = requests.Session()
                session.headers.update({
                    'Authorization': f'Bearer {self.confluence_token}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                })
                
                while True:
                    api_url = f"{self.confluence_url}/rest/api/content/{page_id}/child/page"
                    params = {
                        'expand': 'version,space',
                        'limit': limit,
                        'start': start
                    }
                    
                    response = session.get(api_url, params=params, verify=False, timeout=30)
                    
                    if response.status_code != 200:
                        if attempt < retry_count - 1:
                            print(f"      API error {response.status_code}, retrying...")
                            time.sleep(1)
                            break
                        else:
                            return all_children
                    
                    data = response.json()
                    results = data.get('results', [])
                    
                    if not results:
                        break
                    
                    for child in results:
                        child_info = {
                            'id': child.get('id'),
                            'title': child.get('title'),
                            'url': f"{self.confluence_url}{child.get('_links', {}).get('webui', '')}"
                        }
                        all_children.append(child_info)
                    
                    if len(results) < limit:
                        break
                    
                    start += limit
                    time.sleep(0.1)
                
                if all_children:
                    print(f"      ✅ Found {len(all_children)} children via API")
                return all_children
                
            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"      ⚠️ Error getting children (attempt {attempt + 1}/{retry_count}): {e}")
                    time.sleep(1)
                else:
                    print(f"      ❌ Failed to get children after {retry_count} attempts: {e}")
                    return []
        
        return []
    
    def get_child_pages_from_web(self, url: str, html_content: str) -> List[Dict]:
        """Extract child page links from web scraping with better parsing"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            
            child_links = []
            seen_urls = set()
            
            # Look for various child page patterns
            patterns = [
                ('div', {'class': re.compile(r'(child|page-tree|children)', re.I)}),
                ('ul', {'class': re.compile(r'(child|page-tree|children)', re.I)}),
                ('nav', {'class': re.compile(r'(child|page-tree)', re.I)}),
                ('div', {'id': re.compile(r'(child|children)', re.I)}),
            ]
            
            for tag, attrs in patterns:
                for section in soup.find_all(tag, attrs):
                    for link in section.find_all('a', href=True):
                        href = link.get('href')
                        if href and ('/display/' in href or '/pages/' in href or 'pageId=' in href):
                            full_url = urljoin(base_url, href)
                            
                            # Clean URL
                            full_url = full_url.split('#')[0]  # Remove anchors
                            
                            if full_url not in seen_urls:
                                seen_urls.add(full_url)
                                
                                # Try to extract page ID and title
                                page_id = self.extract_page_id_from_url(full_url)
                                title = link.get_text(strip=True) or "Unknown"
                                
                                child_links.append({
                                    'url': full_url,
                                    'title': title,
                                    'id': page_id
                                })
            
            if child_links:
                print(f"      ✅ Found {len(child_links)} children via web scraping")
            
            return child_links
            
        except Exception as e:
            print(f"      ⚠️ Web scraping error: {e}")
            return []
    
    def fetch_page_content(self, url: str, page_info: Dict, retry_count: int = 2) -> Optional[Dict]:
        """Fetch content from a single Confluence page with retries"""
        
        for attempt in range(retry_count):
            # Try API first
            content = self._fetch_via_api(url, page_info)
            
            if content and content.get('text'):
                return content
            
            # Fallback to web scraping
            content = self._fetch_via_web(url)
            
            if content and content.get('text'):
                return content
            
            if attempt < retry_count - 1:
                print(f"      Retry {attempt + 1}/{retry_count}...")
                time.sleep(1)
        
        return content
    
    def _fetch_via_api(self, url: str, page_info: Dict) -> Optional[Dict]:
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
                api_url = f"{self.confluence_url}/rest/api/content/{page_info['page_id']}"
                params = {'expand': 'body.storage,space,version,title'}
                response = session.get(api_url, params=params, verify=False, timeout=30)
            elif 'space_key' in page_info and 'page_title' in page_info:
                api_url = f"{self.confluence_url}/rest/api/content/search"
                cql_query = f"space='{page_info['space_key']}' AND title~'{page_info['page_title']}'"
                params = {'cql': cql_query, 'expand': 'body.storage,space,version,title', 'limit': 1}
                response = session.get(api_url, params=params, verify=False, timeout=30)
                
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
                page_id = content.get('id')
                
                text = ""
                if content.get('body', {}).get('storage', {}).get('value'):
                    storage_content = content['body']['storage']['value']
                    soup = BeautifulSoup(storage_content, 'html.parser')
                    for tag in soup(['script', 'style', 'meta', 'link']):
                        tag.decompose()
                    text = soup.get_text(separator='\n', strip=True)
                
                return {
                    'title': title,
                    'text': text,
                    'method': 'API',
                    'page_id': page_id,
                    'url': url
                }
            else:
                return None
                
        except Exception as e:
            return None
    
    def _fetch_via_web(self, url: str) -> Optional[Dict]:
        """Fallback to web scraping"""
        try:
            session = requests.Session()
            
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
                    'text': "",
                    'method': 'Web Scraping (Failed)',
                    'html_content': '',
                    'url': url
                }
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Better title extraction
            title = "No title"
            if soup.title:
                title = soup.title.string
            else:
                h1 = soup.find('h1')
                if h1:
                    title = h1.get_text(strip=True)
            
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
            
            if not main_content:
                main_content = soup.body
            
            text = ""
            if main_content:
                for irrelevant in main_content(['script', 'style', 'img', 'input', 'nav', 'header', 'footer']):
                    irrelevant.decompose()
                
                text = main_content.get_text(separator="\n", strip=True)
            
            # Extract page ID from the page itself
            page_id = self.extract_page_id_from_url(url)
            if not page_id:
                # Try to find it in meta tags
                meta_id = soup.find('meta', {'name': 'ajs-page-id'})
                if meta_id:
                    page_id = meta_id.get('content')
            
            return {
                'title': title,
                'text': text,
                'method': 'Web Scraping',
                'html_content': str(response.content),
                'url': url,
                'page_id': page_id
            }
                
        except Exception as e:
            return {
                'title': "Fetch Failed",
                'text': "",
                'method': 'Web Scraping (Failed)',
                'html_content': '',
                'url': url
            }
    
    def analyze_page_tree(self, url: str, mask_data: bool = True, depth: int = 0) -> Optional[Dict]:
        """Recursively analyze page and ALL nested subpages"""
        
        # Prevent infinite loops
        if url in self.visited_pages:
            print(f"{'  ' * depth}⏭️  Already visited")
            return None
            
        if depth > self.max_depth:
            print(f"{'  ' * depth}⚠️  Max depth reached")
            return None
        
        self.visited_pages.add(url)
        self.total_pages += 1
        
        print(f"\n{'  ' * depth}📄 [{self.total_pages}] Depth {depth}: {url}")
        
        # Extract page info
        page_info = self.extract_page_info_from_url(url)
        
        # Fetch content with retries
        content = self.fetch_page_content(url, page_info)
        
        if not content:
            print(f"{'  ' * depth}❌ Failed to fetch")
            self.failed_pages.append(url)
            return None
        
        print(f"{'  ' * depth}✅ {content.get('title', 'No title')[:60]}")
        
        # Track page ID to avoid duplicates
        page_id = content.get('page_id')
        if page_id:
            if page_id in self.visited_page_ids:
                print(f"{'  ' * depth}⏭️  Page ID already processed")
                return None
            self.visited_page_ids.add(page_id)
        
        # Get child pages - try both methods
        child_pages = []
        
        # Method 1: API (most reliable)
        if page_id:
            api_children = self.get_all_child_pages_from_api(page_id)
            child_pages.extend(api_children)
        
        # Method 2: Web scraping (fallback/supplement)
        if content.get('html_content'):
            web_children = self.get_child_pages_from_web(url, content['html_content'])
            # Add web children that aren't already in API children
            existing_ids = {c.get('id') for c in child_pages if c.get('id')}
            for web_child in web_children:
                if web_child.get('id') not in existing_ids:
                    child_pages.append(web_child)
        
        # Mask sensitive data
        if mask_data:
            content['text'] = self.mask_sensitive_data(content['text'])
            content['title'] = self.mask_sensitive_data(content['title'])
        
        # RECURSIVE PROCESSING of children
        children_content = []
        if child_pages:
            print(f"{'  ' * depth}🔍 Processing {len(child_pages)} children...")
            
            for idx, child in enumerate(child_pages, 1):
                print(f"{'  ' * depth}   Child {idx}/{len(child_pages)}: {child.get('title', 'Unknown')[:40]}")
                time.sleep(0.3)
                
                child_url = child.get('url')
                if child_url:
                    # RECURSIVE CALL
                    child_content = self.analyze_page_tree(child_url, mask_data, depth + 1)
                    if child_content:
                        children_content.append(child_content)
        
        content['children'] = children_content
        content['depth'] = depth
        
        return content
    
    def count_total_pages(self, page_tree: Dict) -> int:
        """Count total pages in the tree recursively"""
        if not page_tree:
            return 0
        
        count = 1
        for child in page_tree.get('children', []):
            count += self.count_total_pages(child)
        
        return count
    
    def create_summary_prompt(self, page_tree: Dict) -> tuple:
        """Create comprehensive prompt from page tree"""
        
        def build_content_text(node: Dict, level: int = 0) -> str:
            """Recursively build content text from tree"""
            indent = "  " * level
            text = f"\n{indent}{'#' * (level + 2)} {node.get('title', 'Untitled')}\n\n"
            
            if node.get('text'):
                content = node['text']
                if len(content) > 3000:
                    content = content[:3000] + "\n\n[Content truncated...]"
                text += f"{indent}{content}\n"
            
            for child in node.get('children', []):
                text += build_content_text(child, level + 1)
            
            return text
        
        total_pages = self.count_total_pages(page_tree)
        
        system_prompt = """You are an AI assistant specialized in analyzing comprehensive technical documentation including parent and nested child pages.

Your task is to:
1. Provide an executive summary of the entire page tree
2. Summarize the main page content
3. Summarize each child page and sub-child page hierarchically
4. Identify key themes across all pages
5. Extract important technical details, procedures, or findings
6. Highlight conclusions and recommendations
7. Present everything in clear, structured markdown with proper hierarchy

CRITICAL PRIVACY NOTE: This content contains redacted sensitive information marked as [REDACTED]. Do not attempt to guess or infer redacted content."""
        
        user_prompt = f"""Analyze this Confluence page tree with **{total_pages} total pages** (including all nested subpages).

**Main Page**: {page_tree.get('title', 'Unknown')}
**Total Pages in Tree**: {total_pages}
**Direct Children**: {len(page_tree.get('children', []))}

Please provide a comprehensive analysis covering:
1. Executive Summary
2. Main Page Content
3. Complete Page Hierarchy (all levels)
4. Key Themes and Findings
5. Technical Details
6. Recommendations

Full Content Tree:
{build_content_text(page_tree)}

Remember: Redacted information marked as [REDACTED] should not be inferred or guessed."""
        
        return system_prompt, user_prompt
    
    def summarize_with_gemini(self, page_tree: Dict) -> str:
        """Generate comprehensive summary using Gemini API"""
        
        total_pages = self.count_total_pages(page_tree)
        
        print(f"\n📊 Final Analysis Statistics:")
        print(f"   Total pages discovered: {total_pages}")
        print(f"   Failed pages: {len(self.failed_pages)}")
        print(f"   Main page: {page_tree.get('title', 'Unknown')}")
        print(f"   Direct children: {len(page_tree.get('children', []))}")
        
        try:
            system_prompt, user_prompt = self.create_summary_prompt(page_tree)
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            
            if len(full_prompt) > 1000000:
                print("⚠️  Content very large, truncating...")
                full_prompt = full_prompt[:1000000]
            
            print("🤖 Generating comprehensive summary with Gemini...")
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt
            )
            
            return response.text
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return f"❌ Error calling Gemini API: {e}"
    
    def analyze_confluence_tree(self, url: str, mask_data: bool = True, output_file: str = None) -> str:
        """Main method to analyze complete Confluence page tree and generate summary"""
        
        try:
            print(f"🚀 Confluence Deep Analyzer v3.0 - COMPLETE TREE SCAN")
            print(f"🔒 Privacy mode: {'ENABLED' if mask_data else 'DISABLED'}")
            print(f"📍 Root URL: {url}")
            print("=" * 70)
            
            self.visited_pages.clear()
            self.visited_page_ids.clear()
            self.total_pages = 0
            self.failed_pages = []
            
            start_time = time.time()
            page_tree = self.analyze_page_tree(url, mask_data, depth=0)
            elapsed_time = time.time() - start_time
            
            if not page_tree:
                return "❌ Failed to analyze page tree"
            
            print("\n" + "=" * 70)
            print(f"✅ Analysis complete in {elapsed_time:.1f}s")
            print(f"📊 Total pages: {self.total_pages}")
            print(f"❌ Failed pages: {len(self.failed_pages)}")
            
            if self.failed_pages:
                print("\nFailed URLs:")
                for failed_url in self.failed_pages[:5]:
                    print(f"  - {failed_url}")
                if len(self.failed_pages) > 5:
                    print(f"  ... and {len(self.failed_pages) - 5} more")
            
            summary = self.summarize_with_gemini(page_tree)
            
            total_pages = self.count_total_pages(page_tree)
            
            result = f"""
# Confluence Complete Page Tree Analysis

**Root URL:** {url}
**Main Page:** {page_tree.get('title', 'Unknown')}
**Total Pages Analyzed:** {total_pages}
**Failed Pages:** {len(self.failed_pages)}
**Direct Child Pages:** {len(page_tree.get('children', []))}
**Analysis Time:** {elapsed_time:.1f}s
**Privacy Mode:** {'ENABLED - Sensitive data masked' if mask_data else 'DISABLED'}

---

## Comprehensive Summary

{summary}

---

*Generated by Confluence Deep Analyzer v3.0*
*Total pages in tree: {total_pages}*
*Analysis time: {elapsed_time:.1f} seconds*
"""
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"💾 Saved to: {output_file}")
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Confluence Deep Page Analyzer v3.0 - Complete Tree Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python confluence_summarizer_subpage_scan.py https://confluence.company.com/display/SPACE/Page
    python confluence_summarizer_subpage_scan.py --interactive
    python confluence_summarizer_subpage_scan.py -u URL --output summary.md

v3.0 Improvements:
    ✅ Better retry logic for failed pages
    ✅ Dual child discovery (API + web scraping)
    ✅ Better page ID extraction
    ✅ More detailed logging
    ✅ Handles "No title" pages
        """
    )
    
    parser.add_argument('url', nargs='?', help='Confluence page URL')
    parser.add_argument('-u', '--url', dest='url_flag', help='Confluence page URL')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('--no-mask', action='store_true', help='Disable data masking')
    parser.add_argument('--max-depth', type=int, default=10, help='Max depth (default: 10)')
    parser.add_argument('--version', action='version', version='Confluence Deep Analyzer 3.0')
    
    args = parser.parse_args()
    url = args.url or args.url_flag
    
    try:
        analyzer = ConfluenceDeepAnalyzer()
        
        if args.max_depth:
            analyzer.max_depth = args.max_depth
        
        if args.interactive or not url:
            print("\n" + "="*60)
            print("Confluence Deep Page Analyzer v3.0")
            print("="*60)
            
            while True:
                if not url:
                    url = input("\nEnter URL (or 'quit'): ").strip()
                
                if url.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not url.startswith('http'):
                    print("❌ Invalid URL")
                    url = None
                    continue
                
                summary = analyzer.analyze_confluence_tree(
                    url,
                    mask_data=not args.no_mask,
                    output_file=args.output
                )
                
                print("\n" + "="*60)
                print("SUMMARY")
                print("="*60)
                print(summary)
                
                url = None
        else:
            summary = analyzer.analyze_confluence_tree(
                url,
                mask_data=not args.no_mask,
                output_file=args.output
            )
            
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(summary)
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()