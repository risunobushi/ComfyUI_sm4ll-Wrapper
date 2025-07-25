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

This package provides two nodes for different use cases:

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
- **base_person_mask** (optional): Custom mask for precise control

## Getting API Keys

To use the **Paid API** node, you need a YourMirror API key:

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

1. In ComfyUI, look for either node in the `sm4ll/VTON` category:
   - "sm4ll Wrapper Sampler - Demo Version" (free, rate-limited)
   - "sm4ll Wrapper Sampler - Paid API" (requires API key)

2a. **Input Images**: Set a person image and a product image as inputs (e.g.: using Load Image nodes)

2b. **(Optional) Input Mask**: If you don't want to trust the automasking features, set a Mask input for the person image

2. **Output**: Both nodes output a processed IMAGE that can be connected to a Save Image node, or other nodes for further processing

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

## API Pricing & Limits

### Demo Version (Free)
- **Cost**: Free
- **Rate Limits**: Tied to the same limitations as the HuggingFace Space Demo
- **Queue**: May experience delays during peak usage, queue is shared with all Demo users
- **Non-Commercial License**: Demo image outputs are subjected to a non-commercial license. Check the Terms and Conditions on [studio.yourmirror.io](https://studio.yourmirror.io) for more details.

### Paid API
- **Cost**: $0.10 per API call
- **Rate Limits**: 1000 requests per hour per API key
- **Processing**: Separate queue, separate endpoints for each sm4ll model
- **Billing**: Automatic monthly billing based on actual usage
- **Full Commercial License**: Paid API image outputs are subjected to a full commercial license. Check the Terms and Conditions on [studio.yourmirror.io](https://studio.yourmirror.io) for more details.

## Supported Models & Use Cases

Both nodes support the same AI models:

- **Eyewear**: Sunglasses, prescription glasses, safety goggles
- **Footwear**: Shoes, boots, sneakers, sandals
- **Full-body**: Dresses, coats, full outfits
- **Top Garment**: Shirts, jackets, tops, blouses

## Technical Details

- **API Integration**: 
  - Demo: Uses HuggingFace Spaces Gradio API
  - Paid: Uses production api.yourmirror.io with SSE streaming
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