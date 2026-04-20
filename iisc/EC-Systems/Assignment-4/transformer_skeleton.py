import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import argparse
import os
from collections import OrderedDict

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # TODO: Implement single-head attention
        # Create query, key, value, and output projection layers
        # Remember to set bias=False for query, key, value projections. 
        # Example : self.W_q = nn.Linear(d_model, d_model, bias=False)
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, past_kv=None, use_cache=False):
         # TODO: Implement the forward pass for single-head attention
        # 1. Compute query, key, value projections
        # 2. Handle KV caching if use_cache=True
        # 3. Compute attention scores with scaling #hint use torch.bmm
        # 4. Apply causal masking #hint: use torch.triu
        # 5. Apply softmax to get attention weights
        # 6. Compute context vector #hint: use torch.bmm
        # 7. Compute output and return
        
        # Assuming x is (batch_size, seq_len, d_model)

        q = None
        k = None
        v = None

        # Compute new key and value projections depending on whether we're using cache or not
        if use_cache and past_kv is not None:
            past_k, past_v = past_kv
            q = self.W_q(x)
            # Concatenate the new token row along the rows
            k = torch.cat([past_k, self.W_k(x)], dim=1)
            v = torch.cat([past_v, self.W_v(x)], dim=1)
        else:
            q = self.W_q(x)
            k = self.W_k(x)
            v = self.W_v(x)

        # Use float64 for attention score calculation to reduce numerical issues, then convert back to input dtype for output
        attention_dtype = torch.float64 if q.dtype == torch.float32 else q.dtype
        q64 = q.to(attention_dtype)
        k64 = k.to(attention_dtype)
        v64 = v.to(attention_dtype)

        # Scaling attention scores
        k_t = torch.transpose(k64, 1, 2)
        scaled_scores = torch.bmm(q64, k_t) / self.d_model ** 0.5

        # Causal Masking only if required
        if past_kv is None or not use_cache:
            seq_len = scaled_scores.size(1)
            mask_tensor = torch.tril(torch.ones(seq_len, seq_len, device=scaled_scores.device)) == 0
            scaled_scores = scaled_scores.masked_fill(mask_tensor, float('-inf'))

        # Caclulate softmax and attention context. Return output in float32 for model flow
        attn_sfmax = F.softmax(scaled_scores, dim=-1)
        attn_context = torch.bmm(attn_sfmax, v64)
        attn_output = F.linear(attn_context, self.W_o.weight.to(attention_dtype)).to(x.dtype)
        return attn_output, (k, v) if use_cache else None

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: Implement feed-forward network
        # Create two linear layers 
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # TODO: Implement the forward pass for feed-forward network
        # Two linear layers with ReLU activation in between
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Implement decoder layer
        # Create attention, feed-forward, layer norms, and dropout
        
        self.attention = SingleHeadAttention(d_model=d_model)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, past_kv=None, use_cache=False):
        # TODO: Implement the forward pass for decoder layer
        # 1. Apply layer norm before attention
        # 2. Apply attention followed by dropout
        # 3. Apply layer norm before feed-forward
        # 4. Apply feed-forward then residual connection
        
        # Normalization to the input, Attention block with kv_cache
        residual = x # Temporary store original input to add back to transformer output
        x = self.norm1(x)
        x, past_kv = self.attention(x, past_kv=past_kv, use_cache=use_cache)
        x = self.dropout(x) # Apply dropout to attention output
        x = x + residual # Add the residuals

        # Normalize attn output, send to feed-forward
        residual = x # Store attn output for residual connection
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x) # Apply dropout to FFN output
        x = x + residual # Add the residuals
        
        # Return the final output and the new KV Cache
        return x, past_kv

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # TODO: Implement decoder-only transformer
        # Create token embedding, positional embedding, decoder layers, output projection
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, d_ff=d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, past_kv=None, use_cache=False):
        # TODO: Implement the forward pass for decoder-only transformer
        # 1. Get token embeddings
        # 2. Add positional embeddings (handle position offsets for cached generation)
        # 3. Pass through decoder layers
        # 4. Apply final layer norm
        # 5. Project to vocabulary
        
        # Assuming input_ids is batch_size X seq_len
        
        seq_len = input_ids.shape[1]
        
        # Offset positional embeddings by the previously generated tokens if using cache
        # past_kv is a list of tuples of (k, v) for each layer
        if use_cache and past_kv is not None and past_kv[0] is not None:
            past_seq_len = past_kv[0][0].shape[1] # Get the sequence length from the cached keys
            pos_indices = torch.arange(past_seq_len, past_seq_len + seq_len, device=input_ids.device) # Position indices for the new tokens
        else:
            pos_indices = torch.arange(seq_len, device=input_ids.device) # Geth indices for the input sequence directly
        
        token_embeds = self.token_embedding(input_ids) # Token Embeddings - batch_size X seq_len X d_model
        pos_embeds = self.pos_embedding(pos_indices) # Positional Embeddings - seq_len X d_model
        x = token_embeds + pos_embeds # Add information from both embeddings

        # Execute attention for each attention block, store kv cache for each decode layer
        past_kv_list = [None] * self.num_layers if past_kv is None else past_kv
        for i, decoder in enumerate(self.layers):
            x, past_kv_list[i] = decoder(x, past_kv_list[i], use_cache)

        # Final layer norm in float64 to reduce output drift.
        x = F.layer_norm(
            x.to(torch.float64),
            self.norm.normalized_shape,
            self.norm.weight.to(torch.float64),
            self.norm.bias.to(torch.float64),
            self.norm.eps,
        ).float()
        logits = self.output_projection(x) # Project to vocabulary size - batch_size X seq_len X vocab_size
    
        return logits, past_kv_list
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0, use_cache=True):
        # TODO: Implement the generation method
        # 1. Start with the input sequence
        # 2. Iteratively generate new tokens
        # 3. Use temperature for sampling
        # 4. Use KV caching for efficiency
        
        # Assume input_ids = batch_size X seq_len
        
        past_kv_list = [None] * self.num_layers
        generated_ids = []
        for _ in range(max_new_tokens):
            
            # Compute forward pass. Logits: batch_size X seq_len X vocab_size
            logits, past_kv_list = self.forward(input_ids=input_ids, past_kv=past_kv_list, use_cache=use_cache)
            
            # Slicing to get batch_size X vocab_size
            last_token_logits = logits[:, -1, :]
            # Temperature to introduce randomness, softmax to get the probabilities
            token_prob = F.softmax(last_token_logits / temperature, dim=1) # batch_size X vocab_size
            
            # Get the next token for each batch - batch_size X 1
            next_token = torch.multinomial(token_prob, num_samples=1)
            
            # Use only the new token for the next iteration if using cache, else pass the complete sequence again with the new token appended
            input_ids = next_token if use_cache else torch.cat([input_ids, next_token], dim=1)
            generated_ids.append(next_token)
        
        generated_ids = torch.cat(generated_ids, dim=1) # batch_size X max_new_tokens
        return generated_ids


# ================= DO NOT MODIFY CODE BELOW THIS LINE =================
# Evaluation harness code - do not modify

def load_test_cases(filepath):
    """Load test cases from a file."""
    with open(filepath, 'r') as f:
        test_cases = json.load(f)
    
    # Convert lists back to tensors
    for case in test_cases:
        case['input_ids'] = torch.tensor(case['input_ids'])
        case['expected_logits_no_cache'] = torch.tensor(case['expected_logits_no_cache'])
        case['expected_logits_with_cache'] = torch.tensor(case['expected_logits_with_cache'])
        case['expected_logits_sequential'] = torch.tensor(case['expected_logits_sequential'])
    
    return test_cases

def evaluate_model(model, test_cases, atol=1e-3, with_kv=False):
    """Evaluate model against test cases."""
    model.eval()
    results = []
    
    # logits_no_cache = True
    # with_cache_match = True
    # cache_nocache_match = True

    for i, case in enumerate(test_cases):
        input_ids = case['input_ids']
        expected_logits_no_cache = case['expected_logits_no_cache']
        expected_logits_with_cache = case['expected_logits_with_cache']
        expected_logits_sequential = case['expected_logits_sequential']
        
        with torch.no_grad():
            # Test without caching
            logits_no_cache, _ = model(input_ids, use_cache=False)
            no_cache_match = torch.allclose(logits_no_cache, expected_logits_no_cache, atol=atol)
            
            if with_kv:
                # Test with caching (full sequence)
                logits_with_cache, _ = model(input_ids, use_cache=True)
                with_cache_match = torch.allclose(logits_with_cache, expected_logits_with_cache, atol=atol)

                cache_nocache_match = torch.allclose(logits_no_cache, logits_with_cache, atol=atol)
            
        
        result = {
            'test_case': i + 1,
            'no_cache_match': no_cache_match,
            'with_cache_match': with_cache_match if with_kv else None,
            'cache_nocache_match': cache_nocache_match if with_kv else None,
            'all_match': no_cache_match and (with_cache_match and cache_nocache_match if with_kv else no_cache_match)
        }
        
        if not result['all_match']:
            # Calculate error metrics for debugging
            if not no_cache_match:
                result['no_cache_max_error'] = torch.max(torch.abs(logits_no_cache - expected_logits_no_cache)).item()
            if not with_cache_match:
                result['with_cache_max_error'] = torch.max(torch.abs(logits_with_cache - expected_logits_with_cache)).item()
            if not cache_nocache_match:
                result['cache_nocache_max_error'] = torch.max(torch.abs(logits_no_cache - logits_with_cache)).item()
        
        results.append(result)
    
    # Overall results
    all_passed = all(r['all_match'] for r in results)
    pass_rate = sum(r['all_match'] for r in results) / len(results)
    
    summary = {
        'all_passed': all_passed,
        'pass_rate': pass_rate,
        'num_test_cases': len(test_cases),
        'num_passed': sum(r['all_match'] for r in results),
        'detailed_results': results
    }
    
    return summary

def benchmark_performance(model, input_ids, num_new_tokens=20, use_cache=True, num_runs=3):
    """Benchmark model performance."""
    model.eval()
    
    # Warm-up run
    model.generate(input_ids, num_new_tokens, use_cache=use_cache)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        model.generate(input_ids, num_new_tokens, use_cache=use_cache)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    return avg_time


def main():
    parser = argparse.ArgumentParser(description='Transformer Evaluation Harness')
    parser.add_argument('--mode', type=str, default='run', choices=['generate', 'evaluate', 'kv_evaluate', 'benchmark', 'run'], 
                        help='Mode to run in')
    parser.add_argument('--weights', type=str, default='reference_weights.pt', 
                        help='Path to weights file')
    parser.add_argument('--model_state_dict', type=str, default='model_state_dict.pt', 
                        help='Path to model state dictionary file')
    parser.add_argument('--test_cases', type=str, default='test_cases.json', 
                        help='Path to test cases file')
    parser.add_argument('--vocab_size', type=int, default=1000, 
                        help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=50, 
                        help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=100, 
                        help='Feed-forward dimension')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of decoder layers')
    parser.add_argument('--max_seq_len', type=int, default=128, 
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        # Generate evaluation harness -- not accessible to students
        generate_evaluation_harness(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
    
    elif args.mode == 'evaluate':
        # Evaluate a model
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        if not os.path.exists(args.test_cases):
            print(f"Error: Test cases file {args.test_cases} not found.")
            return
        
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
        
        # Load test cases
        test_cases = load_test_cases(args.test_cases)
        print(f"Test cases loaded from {args.test_cases}")
        
        # Evaluate model
        results = evaluate_model(model, test_cases, with_kv=False)
        
        # Print results
        print(f"Evaluation Results:")
        print(f"  Num test cases: {results['num_test_cases']}")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        # Print result stats - each test case with pass/fail info
        #print(f"  Detailed results: {results['detailed_results']}")

        
        if not results['all_passed']:
            print("\nFailed test cases:")
            for i, result in enumerate(results['detailed_results']):
                if not result['all_match']:
                    print(f"  Test case {result['test_case']}:")
                    if not result.get('no_cache_match', True):
                        print(f"    No cache: Failed (max error: {result.get('no_cache_max_error', 'N/A')})")

    elif args.mode == 'kv_evaluate':
        # Evaluate a model with kv cache against no_kv_cache
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        if not os.path.exists(args.test_cases):
            print(f"Error: Test cases file {args.test_cases} not found.")
            return
        
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
        
        # Load test cases
        test_cases = load_test_cases(args.test_cases)
        print(f"Test cases loaded from {args.test_cases}")
        
        # Evaluate model
        results = evaluate_model(model, test_cases, with_kv=True)
        
        # Print results
        print(f"Evaluation Results:")
        print(f"  Num test cases: {results['num_test_cases']}")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        #detailed results
        #print(f"  Detailed results: {results['detailed_results']}")

        if not results['all_passed']:
            print("\nFailed test cases:")
            for i, result in enumerate(results['detailed_results']):
                if not result['all_match']:
                    print(f"  Test case {result['test_case']}:")
                    if not result.get('no_cache_match', True):
                        print(f"    No cache: Failed (max error: {result.get('no_cache_max_error', 'N/A')})")
                    if not result.get('with_cache_match', True):
                        print(f"    With cache: Failed (max error: {result.get('with_cache_max_error', 'N/A')})")
    
    elif args.mode == 'benchmark':
        # Benchmark model performance
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        

        # Load model state dict
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return

        
        # Create sample input
        input_ids = torch.randint(0, args.vocab_size, (1, 10))
        
        # Benchmark with and without caching
        print("Benchmarking...")
        time_without_cache = benchmark_performance(model, input_ids, use_cache=False)
        time_with_cache = benchmark_performance(model, input_ids, use_cache=True)
        
        print(f"Results:")
        print(f"  Without KV cache: {time_without_cache:.4f} seconds")
        print(f"  With KV cache: {time_with_cache:.4f} seconds")
        print(f"  Speedup: {time_without_cache / time_with_cache:.2f}x")
    
    elif args.mode == 'run':
        # Just a debugging mode

        # Default mode: generate harness if files don't exist, then evaluate and benchmark
        if not os.path.exists('model_state_dict.pt') or not os.path.exists('test_cases.json'):
            generate_evaluation_harness(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        # Load model state dict
        state_dict = torch.load('model_state_dict.pt')

        # Print specific weights from state dict
        #print("From state dict - layer 0 Wq weight:")
        #print(state_dict['layers.0.attention.W_q.weight'])
        #print("From state dict - layer 1 Wk weight:")
        #print(state_dict['layers.1.attention.W_k.weight'])

        try:
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model state dictionary from model_state_dict.pt")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
 
        print("Weights loaded from {}".format(args.model_state_dict))

        # print the structure of state_dict
        #print("State dict structure:")
        #print(state_dict.keys())

        # Verify they're the same
        print("Weights match for layer 0 Wq:", 
        torch.allclose(state_dict['layers.0.attention.W_q.weight'], model.layers[0].attention.W_q.weight))
        print("Weights match for layer 1 Wk:", 
        torch.allclose(state_dict['layers.1.attention.W_k.weight'], model.layers[1].attention.W_k.weight))
        
        # Load test cases
        test_cases = load_test_cases('test_cases.json')
        print(f"Test cases loaded from test_cases.json")

        # Print the first test case
        print("First test case:")
        print(test_cases[0].keys())
        # print the tensor shape for each key
        for key in test_cases[0].keys():
            print(key, test_cases[0][key].shape)

        # Evaluate model
        print("\nEvaluating model...")
        results = evaluate_model(model, test_cases)
        
        # Print evaluation results
        print(f"Evaluation Results:")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        
        # Benchmark
        print("\nBenchmarking performance...")
        input_ids = torch.randint(0, args.vocab_size, (1, 10))
        
        time_without_cache = benchmark_performance(model, input_ids, use_cache=False)
        time_with_cache = benchmark_performance(model, input_ids, use_cache=True)
        
        print(f"Performance Results:")
        print(f"  Without KV cache: {time_without_cache:.4f} seconds")
        print(f"  With KV cache: {time_with_cache:.4f} seconds")
        print(f"  Speedup: {time_with_cache > 0 and time_without_cache / time_with_cache:.2f}x")

if __name__ == "__main__":
    main()