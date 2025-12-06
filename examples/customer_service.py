"""
Demo: Build an RL environment from customer service traces.

This example shows how to:
1. Load trace data
2. Create an RL environment using create_environment()
3. Use the query_agent, tool_mocker, and reward_fn

Run with:
    python examples/customer_service.py
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import create_environment


# Sample customer service traces
SAMPLE_TRACES = [
    {
        "conversation_id": "conv_001",
        "messages": [
            {
                "role": "user",
                "content": "I want to cancel my subscription"
            },
            {
                "role": "assistant",
                "content": "I can help you with canceling your subscription. Let me look up your account first.",
                "tool_calls": [
                    {"name": "lookup_account", "args": {"user_id": "u_12345"}}
                ]
            },
            {
                "role": "tool",
                "name": "lookup_account",
                "content": "",
                "result": {
                    "user_id": "u_12345",
                    "plan": "premium",
                    "status": "active",
                    "billing_date": "2024-01-15"
                }
            },
            {
                "role": "assistant",
                "content": "I found your Premium account. I see your next billing date is January 15th. Before I process the cancellation, may I ask why you'd like to cancel? We may be able to address any concerns."
            },
            {
                "role": "user",
                "content": "It's too expensive for me right now"
            },
            {
                "role": "assistant",
                "content": "I understand budget concerns are important. I can offer you a 50% discount for the next 3 months. Would you like me to apply that instead of canceling?"
            },
            {
                "role": "user",
                "content": "Yes, that would be great!"
            },
            {
                "role": "assistant",
                "content": "Excellent! I've applied the 50% discount to your account. Your next 3 billing cycles will be at the reduced rate. Is there anything else I can help you with?",
                "tool_calls": [
                    {"name": "apply_discount", "args": {"user_id": "u_12345", "discount_percent": 50, "duration_months": 3}}
                ]
            },
            {
                "role": "tool",
                "name": "apply_discount",
                "content": "",
                "result": {"status": "success", "new_monthly_rate": 4.99}
            }
        ],
        "outcome": "resolved",
        "satisfaction_score": 4.8
    },
    {
        "conversation_id": "conv_002",
        "messages": [
            {
                "role": "user",
                "content": "I can't log into my account"
            },
            {
                "role": "assistant",
                "content": "I'm sorry to hear you're having trouble logging in. Let me help you with that. Can you provide the email address associated with your account?"
            },
            {
                "role": "user",
                "content": "It's john.doe@email.com"
            },
            {
                "role": "assistant",
                "content": "Thank you. Let me look up your account.",
                "tool_calls": [
                    {"name": "lookup_by_email", "args": {"email": "john.doe@email.com"}}
                ]
            },
            {
                "role": "tool",
                "name": "lookup_by_email",
                "content": "",
                "result": {
                    "user_id": "u_67890",
                    "email": "john.doe@email.com",
                    "account_status": "locked",
                    "lock_reason": "too_many_attempts"
                }
            },
            {
                "role": "assistant",
                "content": "I found your account. It appears it was temporarily locked due to multiple failed login attempts. I can unlock it for you right now. Would you also like me to send a password reset link?"
            },
            {
                "role": "user",
                "content": "Yes please, unlock it and send me a reset link"
            },
            {
                "role": "assistant",
                "content": "Done! I've unlocked your account and sent a password reset link to john.doe@email.com. Please check your inbox and spam folder. The link will expire in 24 hours. Is there anything else you need help with?",
                "tool_calls": [
                    {"name": "unlock_account", "args": {"user_id": "u_67890"}},
                    {"name": "send_password_reset", "args": {"email": "john.doe@email.com"}}
                ]
            },
            {
                "role": "tool",
                "name": "unlock_account",
                "content": "",
                "result": {"status": "success"}
            },
            {
                "role": "tool",
                "name": "send_password_reset",
                "content": "",
                "result": {"status": "sent", "expires_in": "24h"}
            }
        ],
        "outcome": "resolved",
        "satisfaction_score": 5.0
    }
]


def main():
    """Run the customer service demo."""
    print("=" * 60)
    print("Agent Gym - Customer Service Demo")
    print("=" * 60)
    print()
    
    # Create environment from traces
    print("Creating RL environment from traces...")
    query_agent, tool_mocker, reward_fn = create_environment(SAMPLE_TRACES)
    print("✓ Environment created successfully!")
    print()
    
    # Show available tools learned from traces
    print("Tools learned from traces:")
    available_tools = tool_mocker.__self__.get_available_tools()
    for tool in available_tools:
        print(f"  - {tool}")
    print()
    
    # Demo: Query the agent
    print("-" * 60)
    print("Demo: Querying the agent")
    print("-" * 60)
    
    test_queries = [
        "I want to cancel my subscription",
        "I forgot my password and can't log in",
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        
        try:
            response = query_agent(query)
            print(f"Agent: {response}")
            
            # Score the response
            ground_truth = "I can help you with that. Let me look up your account."
            score = reward_fn(response, ground_truth)
            print(f"Reward Score: {score:.3f}")
        except Exception as e:
            print(f"(Demo mode - Gemini API key required for full functionality)")
            print(f"Error: {e}")
        
        print()
    
    # Demo: Tool mocking
    print("-" * 60)
    print("Demo: Tool Mocking")
    print("-" * 60)
    
    # Mock a tool call
    tool_result = tool_mocker("lookup_account", {"user_id": "u_12345"})
    print(f"\nMocked lookup_account result: {tool_result}")
    
    tool_result = tool_mocker("lookup_by_email", {"email": "test@example.com"})
    print(f"Mocked lookup_by_email result: {tool_result}")
    
    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

