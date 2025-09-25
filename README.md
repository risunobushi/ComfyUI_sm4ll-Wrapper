# ComfyUI_sm4ll-Wrapper
A wrapper node for sm4ll-VTON models with both free demo and paid API access options.

![image](https://github.com/user-attachments/assets/8c08b4fd-a9a6-42ef-a9b8-ec577ac9f0b7)

## Installation

1. Clone or download this repository to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/risunobushi/ComfyUI_sm4ll-Wrapper.git
   ```

2. Install the required dependencies:
   ```bash
   cd ComfyUI_sm4ll-Wrapper
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

## Available Nodes

This package provides three nodes for different use cases:

### 1. **sm4ll Wrapper Sampler - Demo Version** (Free)
Uses the free demo API from HuggingFace Spaces with rate limiting.

**Configure Inputs**:
- **base_person_image**: The person/model image (required)
- **product_image**: The garment/product to try on (required)  
- **model_choice**: Select from "eyewear", "footwear", "full-body", or "top garment" (required)
- **base_person_mask** (optional): Custom mask if you don't trust the automasking feature

### 2. **sm4ll Wrapper Sampler - Paid API** (Requires API Key)
Uses the production API with higher reliability, no rate limits, and priority processing.

**Configure Inputs**:
- **base_person_image**: The person/model image (required)
- **product_image**: The garment/product to try on (required)  
- **model_choice**: Select from "eyewear", "footwear", "full-body", or "top garment" (required)
- **api_key**: Your YourMirror API key (required) - format: `ym_29_characters`
- **quality_tier**: Select quality level - "normal" (16 steps) or "high" (40 steps) (optional, defaults to "normal")
- **base_person_mask** (optional): Custom mask for precise control

### 3. **sm4ll Wrapper Lookbook Sampler** (Requires API Key)
Create multi-garment lookbook compositions with a single person image and up to 4 garment images.

**Configure Inputs**:
- **person_image**: The person/model image (required)
- **garment_1**: First garment to include in the composition (required)
- **garment_2**: Second garment (optional)
- **garment_3**: Third garment (optional)
- **garment_4**: Fourth garment (optional)
- **mode**: Gender mode - "Male" or "Female" (required)
- **quality**: Quality level - "Normal" or "High" (required)
- **prompt**: Optional text description to guide the generation (optional)
- **api_key**: Your YourMirror API key (required) - format: `ym_29_characters`

**Note**: The Lookbook node composes multiple garments into a grid layout and generates a comprehensive try-on visualization showing the person wearing all provided garments simultaneously.

## Getting API Keys

To use the **Paid API** or **Lookbook** nodes, you need a YourMirror API key:

1. **Sign up** at [studio.yourmirror.io](https://studio.yourmirror.io)
2. **Complete the onboarding** confirm your email and verify your account
3. **Navigate to API Keys** section in your dashboard
4. **Create a new API key** - you'll see it only once, so copy it immediately
5. **Use the API key** in the ComfyUI node (format: `ym_` followed by 29 characters)

### API Key Features:
- **Format**: `ym_` + 29 random characters (total length: 32)
- **Rate Limit**: 1000 requests per hour per key
- **Usage Tracking**: All API calls are logged for billing and analytics
- **NSFW Filtering**: Automatic content screening for appropriate use

## Usage Instructions

1. In ComfyUI, look for any of the three nodes in the `sm4ll/VTON` category:
   - "sm4ll Wrapper Sampler - Demo Version" (free, rate-limited)
   - "sm4ll Wrapper Sampler - Paid API" (requires API key)
   - "sm4ll Wrapper Lookbook Sampler" (requires API key, multi-garment)

2a. **Input Images**:
   - For Demo/Paid API nodes: Set a person image and a product image as inputs (e.g.: using Load Image nodes)
   - For Lookbook node: Set a person image and 1-4 garment images as inputs

2b. **(Optional) Input Mask**: If you don't want to trust the automasking features, set a Mask input for the person image (Demo/Paid API nodes only)

3. **Additional Configuration**:
   - For Lookbook node: Select gender mode (Male/Female), quality level, and optionally add a text prompt

4. **Output**: All nodes output a processed IMAGE that can be connected to a Save Image node, or other nodes for further processing

## How It Works

### Demo Version (Free)
1. **Image Processing**: Input images are automatically resized to 1.62 megapixels using Lanczos interpolation
2. **Upload**: Images are uploaded to the HuggingFace Spaces demo environment's temp storage for handling
3. **API Call**: Uses the free demo API with rate limiting and queue management
4. **Result Retrieval**: Downloads the processed image and converts back to ComfyUI tensor format

### Paid API Version
1. **Image Processing**: Same high-quality image preprocessing as demo version
2. **Authentication**: Validates your API key format and permissions
3. **Upload**: Images are uploaded to the production API infrastructure
4. **Priority Processing**: Dedicated endpoints for faster processing

### Lookbook Version (Multi-Garment)
1. **Image Processing**: All images (person + up to 4 garments) are resized to 1.62 megapixels
2. **Garment Composition**: Multiple garment images are composed into a grid layout on the backend
3. **Authentication**: Validates your API key format and permissions
4. **Upload**: All images are uploaded to the production API infrastructure
5. **Lookbook Generation**: Uses specialized AI model to create comprehensive try-on visualization
6. **Mode-Aware Processing**: Optimized for male or female body types based on selection

## API Pricing & Limits

### Demo Version (Free)
- **Cost**: Free
- **Rate Limits**: Tied to the same limitations as the HuggingFace Space Demo
- **Queue**: May experience delays during peak usage, queue is shared with all Demo users
- **Non-Commercial License**: Demo image outputs are subjected to a non-commercial license. Check the Terms and Conditions on [studio.yourmirror.io](https://studio.yourmirror.io) for more details.

### Paid API (Single Garment VTON)
- **Cost (Normal Quality)**: $0.05 per API call (16 sampling steps)
- **Cost (High Quality)**: $0.10 per API call (40 sampling steps)
- **Rate Limits**: 1000 requests per hour per API key
- **Processing**: Separate queue, separate endpoints for each sm4ll model
- **Billing**: Automatic monthly billing based on actual usage
- **Full Commercial License**: Paid API image outputs are subjected to a full commercial license. Check the Terms and Conditions on [studio.yourmirror.io](https://studio.yourmirror.io) for more details.

### Lookbook API (Multi-Garment)
- **Cost (Normal Quality)**: $0.15 per API call (3 credits)
- **Cost (High Quality)**: $0.30 per API call (6 credits)
- **Rate Limits**: 1000 requests per hour per API key
- **Processing**: Dedicated lookbook processing pipeline
- **Features**: Supports 1-4 garments, gender-specific optimization, text prompts
- **Billing**: Automatic monthly billing based on actual usage
- **Full Commercial License**: Lookbook API image outputs are subjected to a full commercial license. Check the Terms and Conditions on [studio.yourmirror.io](https://studio.yourmirror.io) for more details.

## Supported Models & Use Cases

### Demo & Paid API Nodes (Single Garment)
Support specific model types for single-garment try-on:

- **Eyewear**: Sunglasses, prescription glasses, safety goggles
- **Footwear**: Shoes, boots, sneakers, sandals
- **Full-body**: Dresses, coats, full outfits
- **Top Garment**: Shirts, jackets, tops, blouses

### Lookbook Node (Multi-Garment)
Uses specialized lookbook VTON model that can handle any combination of garment types:

- **Mixed Categories**: Combine different garment types (shoes + dress + accessories)
- **Multiple Items**: Up to 4 different garments in one composition
- **Cross-Category**: No restrictions on garment combinations
- **Gender-Optimized**: Separate processing pipelines for male/female body types

## Technical Details

- **API Integration**:
  - Demo: Uses HuggingFace Spaces Gradio API
  - Paid API: Uses production api.yourmirror.io with SSE streaming
  - Lookbook: Uses production api.yourmirror.io with direct HTTP POST
- **Image Requirements**: 
  - Supported formats: JPG, PNG, WEBP
  - Optimal size: Around 1.62 megapixels (automatically resized)

## Troubleshooting

### Common Issues

- **Red Output Image**: Indicates an error occurred during processing. Check the console for detailed error messages.
- **API Key Errors**: Ensure your API key starts with `ym_` and is exactly 32 characters long
- **Upload Failures**: Check your internet connection and image format compatibility
- **Timeout Issues**: 
  - Demo: May experience delays during peak usage
  - Paid API: 5-minute timeout for processing, contact support if persistent

### Error Messages

- `"API key is required"`: You're using the Paid API node without providing an API key
- `"Invalid API key format"`: Your API key should be `ym_` followed by 29 characters
- `"Failed to upload"`: Network connectivity or server availability issue

## Requirements

- torch
- torchvision  
- numpy
- Pillow
- requests

## Credits

- Built for the [sm4ll-VTON](https://huggingface.co/spaces/sm4ll-VTON/sm4ll-VTON-Demo) family of models