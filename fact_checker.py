from duckduckgo_search import DDGS
from typing import List, Dict, Any
import datetime

class NewsVerifier:
    """
    Handles external verification of anomalies using real-time web search.
    """
    
    def verify_anomaly(self, district: str, date_obj: datetime.datetime, issue: str) -> List[Dict[str, Any]]:
        """
        Queries news sources for events matching the district and issue within the relevant timeframe.
        """
        date_str = date_obj.strftime('%B %Y')
        query = f"{district} Uttarakhand {issue} {date_str} news"
        
        try:
            # Execute search with a limit to ensure speed
            results = DDGS().text(query, max_results=3)
            
            clean_links = []
            if results:
                for res in results:
                    clean_links.append({
                        "title": res.get('title', 'Untitled'),
                        "link": res.get('href', '#'),
                        "snippet": res.get('body', '')
                    })
            return clean_links

        except Exception as e:
            print(f"[NewsVerifier Error]: {e}")
            return []