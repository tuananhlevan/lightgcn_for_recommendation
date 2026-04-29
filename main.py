import argparse
import sys
import os

# Add the src directory to the Python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train
from src.inference import recommend_for_user

def main():
    parser = argparse.ArgumentParser(description="LightGCN Recommendation System")
    
    parser.add_argument('--train', action='store_true', help="Run the training pipeline")
    parser.add_argument('--infer', type=int, metavar='USER_ID', help="Generate recommendations for a specific User ID")
    parser.add_argument('--top_k', type=int, default=5, help="Number of recommendations to generate (default: 5)")
    
    args = parser.parse_args()
    
    if args.train:
        print("Initializing Training Pipeline...")
        train()
        
    elif args.infer is not None:
        print(f"Initializing Inference for User {args.infer}...")
        recommend_for_user(raw_user_id=args.infer, top_k=args.top_k)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()