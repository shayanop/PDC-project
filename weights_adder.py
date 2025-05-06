



import random
import argparse

def add_random_weights(input_file, output_file, min_weight=1, max_weight=10):
    """
    Add random weights to edges in an undirected graph.
   
    Args:
        input_file (str): Path to the input file containing edge list.
        output_file (str): Path to the output file for weighted edges.
        min_weight (int): Minimum possible weight value.
        max_weight (int): Maximum possible weight value.
    """
    # Read edges from the input file
    with open(input_file, 'r') as f:
        edges = [line.strip().split() for line in f if line.strip()]
   
    # Add random weights to edges
    weighted_edges = []
    for u, v in edges:
        weight = random.randint(min_weight, max_weight)
        weighted_edges.append((u, v, weight))
   
    # Write weighted edges to the output file
    with open(output_file, 'w') as f:
        for u, v, w in weighted_edges:
            f.write(f"{u}\t{v}\t{w}\n")
   
    print(f"Successfully added random weights to {len(edges)} edges.")
    print(f"Weighted graph saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add random weights to an undirected graph.')
    parser.add_argument('input_file', type=str, help='Path to the input file containing edge list')
    parser.add_argument('output_file', type=str, help='Path to the output file for weighted edges')
    parser.add_argument('--min_weight', type=int, default=1, help='Minimum possible weight value')
    parser.add_argument('--max_weight', type=int, default=10, help='Maximum possible weight value')
   
    args = parser.parse_args()
   
    add_random_weights(args.input_file, args.output_file, args.min_weight, args.max_weight)

