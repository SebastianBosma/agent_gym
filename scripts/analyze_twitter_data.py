"""
Quick analytics on the Twitter Customer Support dataset.
Analyzes thread lengths, agent distribution, and potential tool usage patterns.
"""

import pandas as pd
from collections import defaultdict, Counter
import re

def load_data(filepath: str) -> pd.DataFrame:
    """Load the Twitter customer support CSV."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} tweets\n")
    return df

def analyze_thread_lengths(df: pd.DataFrame) -> dict:
    """
    Analyze conversation thread lengths.
    A thread is a chain of tweets linked by in_response_to_tweet_id.
    """
    print("=" * 60)
    print("THREAD LENGTH ANALYSIS")
    print("=" * 60)
    
    # Build response graph
    response_to = {}
    for _, row in df.iterrows():
        tweet_id = row['tweet_id']
        parent = row['in_response_to_tweet_id']
        if pd.notna(parent):
            # Handle multiple parents (comma-separated)
            parents = str(parent).split(',')
            response_to[tweet_id] = int(float(parents[0].strip()))
    
    # Find thread roots (tweets with no parent or parent not in dataset)
    all_tweet_ids = set(df['tweet_id'])
    roots = set()
    for tweet_id in all_tweet_ids:
        # Trace back to root
        current = tweet_id
        while current in response_to and response_to[current] in all_tweet_ids:
            current = response_to[current]
        roots.add(current)
    
    # Build threads from roots
    children = defaultdict(list)
    for child, parent in response_to.items():
        if parent in all_tweet_ids:
            children[parent].append(child)
    
    # Calculate thread lengths using BFS
    thread_lengths = []
    for root in roots:
        # BFS to find max depth
        queue = [(root, 1)]
        max_depth = 1
        while queue:
            node, depth = queue.pop(0)
            max_depth = max(max_depth, depth)
            for child in children.get(node, []):
                queue.append((child, depth + 1))
        thread_lengths.append(max_depth)
    
    # Statistics
    thread_lengths = sorted(thread_lengths)
    total_threads = len(thread_lengths)
    
    print(f"Total conversation threads: {total_threads:,}")
    print(f"\nThread length distribution:")
    print(f"  Min:    {min(thread_lengths)}")
    print(f"  Max:    {max(thread_lengths)}")
    print(f"  Mean:   {sum(thread_lengths)/len(thread_lengths):.2f}")
    print(f"  Median: {thread_lengths[len(thread_lengths)//2]}")
    
    # Histogram buckets
    buckets = Counter()
    for length in thread_lengths:
        if length == 1:
            buckets["1 (single tweet)"] += 1
        elif length <= 3:
            buckets["2-3"] += 1
        elif length <= 5:
            buckets["4-5"] += 1
        elif length <= 10:
            buckets["6-10"] += 1
        else:
            buckets["11+"] += 1
    
    print(f"\nThread length buckets:")
    for bucket in ["1 (single tweet)", "2-3", "4-5", "6-10", "11+"]:
        count = buckets.get(bucket, 0)
        pct = count / total_threads * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket:20s}: {count:6,} ({pct:5.1f}%) {bar}")
    
    return {"lengths": thread_lengths, "buckets": dict(buckets)}

def analyze_agent_distribution(df: pd.DataFrame) -> dict:
    """Analyze distribution of support agents (companies) in the dataset."""
    print("\n" + "=" * 60)
    print("AGENT (COMPANY) DISTRIBUTION")
    print("=" * 60)
    
    # Filter to only agent responses (inbound=False)
    agent_tweets = df[df['inbound'] == False]
    print(f"Total agent tweets: {len(agent_tweets):,}")
    
    # Count by author_id (company name)
    agent_counts = agent_tweets['author_id'].value_counts()
    
    print(f"Unique agents (companies): {len(agent_counts)}")
    print(f"\nTop 20 agents by tweet count:")
    
    for i, (agent, count) in enumerate(agent_counts.head(20).items()):
        pct = count / len(agent_tweets) * 100
        bar = "█" * int(pct)
        print(f"  {i+1:2}. {agent:25s}: {count:6,} ({pct:5.1f}%) {bar}")
    
    # Show remaining
    remaining = agent_counts[20:].sum() if len(agent_counts) > 20 else 0
    if remaining > 0:
        print(f"  ... {len(agent_counts) - 20} more agents with {remaining:,} tweets total")
    
    return {"agent_counts": agent_counts.to_dict()}

def analyze_tool_patterns(df: pd.DataFrame) -> dict:
    """
    Look for potential 'tool-like' patterns in agent responses.
    These might include: URLs, DM requests, phone numbers, specific instructions.
    """
    print("\n" + "=" * 60)
    print("TOOL-LIKE PATTERNS IN AGENT RESPONSES")
    print("=" * 60)
    
    agent_tweets = df[df['inbound'] == False]
    total_agent = len(agent_tweets)
    
    patterns = {
        "URL/Link shared": r'https?://\S+',
        "DM request": r'\b(DM|direct message|send.{0,20}message)\b',
        "Phone number": r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{4,5}[-.\s]?\d{6}\b',
        "Email mention": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "Account/Settings instruction": r'\b(settings|account|profile|preferences)\b',
        "Step-by-step (numbered)": r'\b[1-9]\.\s+\w+',
        "Apology/Sorry": r'\b(sorry|apolog|regret)\b',
        "Escalation mention": r'\b(escalat|supervisor|manager|team)\b',
    }
    
    results = {}
    print(f"Analyzing {total_agent:,} agent tweets for tool-like patterns:\n")
    
    for pattern_name, regex in patterns.items():
        matches = agent_tweets['text'].str.contains(regex, case=False, na=False)
        count = matches.sum()
        pct = count / total_agent * 100
        bar = "█" * int(pct / 2)
        print(f"  {pattern_name:30s}: {count:6,} ({pct:5.1f}%) {bar}")
        results[pattern_name] = {"count": int(count), "percentage": pct}
    
    # Sample some tweets with URLs (common tool-like behavior)
    print("\n" + "-" * 60)
    print("Sample agent tweets with URLs (potential tool usage):")
    print("-" * 60)
    url_tweets = agent_tweets[agent_tweets['text'].str.contains(r'https?://\S+', na=False)]
    for _, row in url_tweets.head(5).iterrows():
        text = row['text'][:150] + "..." if len(str(row['text'])) > 150 else row['text']
        print(f"\n  [{row['author_id']}]")
        print(f"  {text}")
    
    return results

def main():
    # Use full dataset
    filepath = "/Users/sebastianbosma/Documents/agent_gym/data/raw/twitter_support/twcs.csv"
    
    df = load_data(filepath)
    
    # Basic info
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Columns: {list(df.columns)}")
    print(f"Total tweets: {len(df):,}")
    print(f"Inbound (customer) tweets: {(df['inbound'] == True).sum():,}")
    print(f"Outbound (agent) tweets: {(df['inbound'] == False).sum():,}")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    
    # Run analyses
    thread_analysis = analyze_thread_lengths(df)
    agent_analysis = analyze_agent_distribution(df)
    tool_analysis = analyze_tool_patterns(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: SUITABILITY FOR AGENT TRAINING")
    print("=" * 60)
    print("""
✓ PROS:
  - Large dataset with real customer service interactions
  - Multi-turn conversations (threads up to 10+ turns)
  - Diverse set of companies/agents with different styles
  - Natural language patterns (apologies, escalations, instructions)
  - Some tool-like behavior (URLs, DM requests, phone numbers)

⚠ CONSIDERATIONS:
  - No explicit tool calls (it's Twitter, not an API-based system)
  - Tool usage would need to be inferred or synthesized
  - Thread reconstruction required (data is flattened)
  - Some single-tweet "threads" (no conversation context)
    """)

if __name__ == "__main__":
    main()

