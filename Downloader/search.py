import pandas as pd
import asf_search as asf
import argparse 
import os, sys
from .wkt_areas import WKTS

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

    def search(self, max_res=10):
        # Set search parameters
        search_parameters = {
            "platform": "Sentinel-1",
            "processingLevel": "RAW",
            "intersectsWith": self.wkt,
            "maxResults": max_res,
            "beamSwath": ["IW"],
            # "beamSwath": ["S1", "S2", "S3", "S4", "S5", "S6"], # Stripmap mode
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
    
def download_single(wkt, start, end, username, psw, output_dir, download=False, max_res=2):
        D = RAWHandler(wkt, start, end)
        df = D.search(max_res=max_res)
        print('='*50)
        print('Search results:', df)
        print('='*50)
        if download and len(df) > 0:
            print('Downloading...')
            os.makedirs(output_dir, exist_ok=True)
            D.download(username, psw, output_dir=output_dir)
            print('='*50)
            print('Download complete!')
        return df
    
if __name__ == "__main__":
    """
    Example usage:
        python -m Downloader.search --all --start "2020-01-01" --end "2022-12-31" --out "./Data/RAW/IW/" --download --username "username" --psw "password" --max 20
    """    
    # Set search parameters with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wkt", help="WKT polygon to search", default='POLYGON((11.859924690893369 41.87951862335169,12.679779427221494 41.87951862335169,12.679779427221494 41.361079422445385,11.859924690893369 41.361079422445385,11.859924690893369 41.87951862335169))')
    parser.add_argument("--start", help="Start date", default='2020-01-01')
    parser.add_argument("--end", help="End date", default='2020-12-31')
    parser.add_argument("--out", help="Output folder of products", default='./Data/RAW/IW/')
    parser.add_argument("--max_res", help="Maximum number of results", default=10)
    parser.add_argument('--download', dest='download', action='store_true', help='Set to download mode')
    parser.add_argument('--all', dest='download_all', action='store_true', help='Set to all products')

    parser.add_argument("--username", help="ASF username", type=str, default=None)
    parser.add_argument("--psw", help="ASF password", type=str, default=None)
    
    args = parser.parse_args()

    if args.username is not None:
        username = args.username
        assert args.psw is not None, "Password not provided."
        psw = args.psw
        
    else:
        username, psw = os.environ["USERNAME"], os.environ["PASSWORD"]
        print('Username:', username)
        print('Password:', psw)

    if args.download_all:
        wkt_areas = WKTS
        df_list = []
        for key in wkt_areas:
            print('Downloading product for:', key)
            regional_out = os.path.join(args.out, key)
            df = download_single(wkt=wkt_areas[key], start=args.start, end=args.end, username=username, psw=psw, output_dir=regional_out, download=args.download, max_res=args.max_res)
            df_list.append(df)
        df = pd.concat(df_list)
        df.to_csv(os.path.join(args.out, 'all_products.csv'), index=False)
    else:
        df = download_single(wkt=args.wkt, start=args.start, end=args.end, username=args.username, psw=args.psw, output_dir=args.out, download=args.download, max_res=args.max_res)
        df.to_csv(os.path.join(args.out, 'all_products.csv'), index=False)

          
    
    