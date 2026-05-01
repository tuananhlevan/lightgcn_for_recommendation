import pandas as pd

import argparse

def convert_ml_1m_to_csv(source_dir='.', output_dir='.'):
    # Define column names based on MovieLens 1M documentation
    ratings_cols = ['userId', 'movieId', 'rating', 'timestamp']
    users_cols = ['userId', 'gender', 'age', 'occupation', 'zip_code']
    movies_cols = ['movieId', 'title', 'genres']

    # 1. Process Ratings
    print("Converting ratings.dat...")
    ratings = pd.read_csv(f'{source_dir}/ratings.dat', 
                          sep='::', 
                          engine='python', 
                          names=ratings_cols, 
                          encoding='latin-1')
    ratings.to_csv(f'{output_dir}/ratings.csv', index=False)

    # 2. Process Users
    print("Converting users.dat...")
    users = pd.read_csv(f'{source_dir}/users.dat', 
                        sep='::', 
                        engine='python', 
                        names=users_cols, 
                        encoding='latin-1')
    users.to_csv(f'{output_dir}/users.csv', index=False)

    # 3. Process Movies
    print("Converting movies.dat...")
    movies = pd.read_csv(f'{source_dir}/movies.dat', 
                         sep='::', 
                         engine='python', 
                         names=movies_cols, 
                         encoding='latin-1')
    movies.to_csv(f'{output_dir}/movies.csv', index=False)

    print("Conversion complete! Check your output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--source', type=str, help="Source destination of raw data")
    parser.add_argument('--output', type=str, help="Output destination of processed data")
    
    args = parser.parse_args()
    
    convert_ml_1m_to_csv(args.source, args.output)