from transformers import pipeline, logging
import pandas as pd
import torch

# Clear GPU cache and setup logging
torch.cuda.empty_cache()
logging.set_verbosity_info()

# Initialize generator pipeline
generator = pipeline(
    'text-generation',
    model='gpt2',
    device=0,
    truncation=True,
    max_new_tokens=50,
    batch_size=1,
    pad_token_id=50256  # Set padding token explicitly
)


def generate_examples(base_prompt, example_queries, num_examples=15):
    prompt = f"""Task: Generate a customer service questions asked by customers to a mattresses selling company.
Topic: {base_prompt}
Rules: 
- Question should be less than 10 words
- Must be related to {base_prompt}
- Should be similar to these examples:
{example_queries}

Generate a question:"""

    examples = []
    while len(examples) < num_examples:
        try:
            result = generator(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
            generated = result[0]['generated_text']
            if len(generated.split()) <= 10 and '?' in generated:
                examples.append(generated)
        except Exception as e:
            print(f"Error: {e}")
            continue
    return examples


# Example prompts with sample queries
class_examples = {
    'Monthly installment': {
        'prompt': " Equated Monthly Installment EMI and payment options for mattress purchase",
        'examples': """
- Do you have EMI options?
- I want in installment
- Down payments"""
    },
    'COD': {
        'prompt': "Cash on Delivery or COD payment for mattress",
        'examples': """
- Is COD available?
- Do you accept cash on delivery?
- Can I pay later on delivery """
    },
    'ORTHO_FEATURES': {
        'prompt': "Orthopedic features and benefits of mattress",
        'examples': """
- Is it good for back pain?
- What orthopedic features does it have?
- Does it support spine alignment?"""
    },
    'ERGO_FEATURES': {
        'prompt': "Ergonomic features and comfort of mattress",
        'examples': """
- What ergonomic features are included?
- How does it support body posture?
- Tell me about comfort features"""
    },
    'COMPARISON': {
        'prompt': "Compare different mattress models",
        'examples': """
- Which available model is better?
- Compare soft vs firm mattress
- What's the difference between your available models?"""
    },
    'WARRANTY': {
        'prompt': "Warranty terms and coverage for mattress",
        'examples': """
- What's the warranty period?
- What's covered in warranty?
- How long is the warranty valid?"""
    },
    '100_NIGHT_TRIAL_OFFER': {
        'prompt': "100-night trial period for mattress",
        'examples': """
- How does trial period work?
- Can I return during trial?
- What's the trial period policy?"""
    },
    'SIZE_CUSTOMIZATION': {
        'prompt': "Custom size options for mattress",
        'examples': """
- Can you make custom size?
- Is size customization possible?
- Do you make special sizes?"""
    },
    'WHAT_SIZE_TO_ORDER': {
        'prompt': "Help choosing mattress size",
        'examples': """
- Which size should I buy?
- What size fits my bed?
- Recommend size for couple?"""
    },
    'LEAD_GEN': {
        'prompt': "Request for mattress information and contact",
        'examples': """
- Please contact me about details
- Need more information
- Request callback for details"""
    },
    'CHECK_PINCODE': {
        'prompt': "Delivery availability check by pincode address",
        'examples': """
- Do you deliver to my pincode?
- Check delivery availability
- Is delivery possible here?"""
    },
    'DISTRIBUTORS': {
        'prompt': "Find nearby stores and distributors",
        'examples': """
- Where is your nearest store?
- Dealer in my city?
- Local distributor contact?"""
    },
    'MATTRESS_COST': {
        'prompt': "Price and cost information for mattress",
        'examples': """
- What's the price?
- How much for queen size?
- Cost of single mattress?"""
    },
    'PRODUCT_VARIANTS': {
        'prompt': "Different types and variants of mattresses",
        'examples': """
- What types are available?
- Different variants you have?
- Show all mattress options"""
    },
    'ABOUT_SOF_MATTRESS': {
        'prompt': "Information about SOF mattress company",
        'examples': """
- Tell me about your company
- Company background details?
- About SOF mattress brand?"""
    },
    'DELAY_IN_DELIVERY': {
        'prompt': "Delayed delivery inquiries",
        'examples': """
- Why is delivery delayed?
- When will order arrive?
- Reason for delivery delay?"""
    },
    'ORDER_STATUS': {
        'prompt': "Check status of mattress order",
        'examples': """
- Where is my order?
- Track order status
- Current delivery status?"""
    },
    'RETURN_EXCHANGE': {
        'prompt': "Return and exchange policies",
        'examples': """
- How to return mattress?
- What's the return policy?
- Exchange possible?"""
    },
    'CANCEL_ORDER': {
        'prompt': "Cancel mattress order",
        'examples': """
- How to cancel order?
- Cancel my booking
- Order cancellation process?"""
    },
    'PILLOWS': {
        'prompt': "Information about pillows",
        'examples': """
- Do you sell pillows?
- Pillow types available?
- Price of pillows?"""
    },
    'OFFERS': {
        'prompt': "Current offers and discounts",
        'examples': """
- Any current offers?
- What discounts available?
- Ongoing promotional deals?"""
    }
}

# Generate data
generated_data = []
for label, content in class_examples.items():
    print(f"\nGenerating for {label}...")
    examples = generate_examples(content['prompt'], content['examples'])

    for example in examples:
        generated_data.append({
            'sentence': example,
            'label': label
        })

# Save results
df = pd.DataFrame(generated_data)
df.to_csv('llm_generated_queries.csv', index=False)

# Print statistics
print(f"\nTotal generated queries: {len(df)}")
print("\nQueries per label:")
print(df['label'].value_counts())
