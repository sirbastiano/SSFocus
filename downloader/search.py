import pandas as pd
from io import StringIO
import asf_search as asf
import argparse 
import os, sys

class RAWHandler:
    def __init__(self, wkt, start, end):
        # Initialize the object and set the values of the attributes
        self.wkt = wkt        
        self.start = self.date_check(start)
        self.end = self.date_check(end)
        self.results = None
    
    @staticmethod
    def date_check(date):
        if not date.endswith('Z'):
            if date.endswith('T00:00:00.000Z'):
                return date
            else:
                return date + 'T00:00:00.000Z'
        else:
            return date

    def search(self):
        # Set search parameters
        search_parameters = {
            "platform": "Sentinel-1",
            "processingLevel": "RAW",
            "intersectsWith": self.wkt,
            "maxResults": 1,
            'start': self.start,
            'end': self.end,
        }

        try:
            results = asf.geo_search(**search_parameters)
            self.results = results
        except Exception:
            print("No results found.")
            sys.exit(1)

        with open("search_results.csv", "w") as f:
            f.writelines(results.csv())
        
        df = pd.read_csv("search_results.csv")
        os.remove("search_results.csv")
        return df
    
    def download(self, username, psw, output_dir):
        session = asf.ASFSession().auth_with_creds(username, psw)
        try:
            self.results.download(path=output_dir, session=session)
        except Exception as e:
            print('Error downloading results:', e)
            sys.exit(1)
    
    
if __name__ == "__main__":    
    # Set search parameters with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wkt", help="WKT polygon to search", default='POLYGON((11.859924690893369 41.87951862335169,12.679779427221494 41.87951862335169,12.679779427221494 41.361079422445385,11.859924690893369 41.361079422445385,11.859924690893369 41.87951862335169))')
    parser.add_argument("--start", help="Start date", default='2020-01-01')
    parser.add_argument("--end", help="End date", default='2020-12-31')
    
    parser.add_argument("--username", help="ASF username", type=str)
    parser.add_argument("--psw", help="ASF password", type=str)
    
    args = parser.parse_args()
    
    D = RAWHandler(args.wkt, args.start, args.end)
    df = D.search()
    print('='*50)
    print('Search results:', df)
    print('='*50)
    
    print('Downloading...')
    D.download(args.username, args.psw, output_dir='/Users/robertodelprete/Documents/PyScripts/SARLens/SSFocus/Data/RAW')
    print('='*50)
    print('Download complete!')
    
          
    
    