# ComfyUI_sm4ll-Wrapper
A wrapper node for sm4ll-VTON models: https://huggingface.co/spaces/sm4ll-VTON/sm4ll-VTON-Demo

# ComfyUI VTON API Node

A ComfyUI custom node that integrates with the sm4ll-VTON API for virtual try-on functionality.

## Features

- **Virtual Try-On**: Generate realistic virtual try-on images using AI
- **Multiple Models**: Support for eyewear, footwear, and full-body garments
- **Smart Image Processing**: Automatic resize to 1.62 megapixels using Lanczos interpolation
- **Direct Integration**: Seamlessly upload images to Gradio space and retrieve results
- **Error Handling**: Robust error handling with fallback placeholder images

## Installation

1. Clone or download this repository to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/your-repo/ComfyUI_sm4ll-Wrapper.git
   ```

2. Install the required dependencies:
   ```bash
   cd ComfyUI_sm4ll-Wrapper
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

## Usage

1. **Add the Node**: In ComfyUI, look for "VTON API Node" in the `sm4ll/VTON` category

2. **Configure Inputs**:
   - **base_person_image**: The person/model image (required)
   - **product_image**: The garment/product to try on (required)  
   - **model_choice**: Select from "eyewear", "footwear", or "full-body" (required)
   - **gradio_space_url**: The Gradio space URL (optional, defaults to sm4ll-VTON demo)

3. **Connect Output**: The node outputs a processed IMAGE that can be connected to other nodes

## How It Works

1. **Image Processing**: Input images are automatically resized to 1.62 megapixels using Lanczos interpolation for optimal results
2. **Upload**: Images are uploaded directly to the Gradio space's internal file system
3. **API Call**: The VTON API is called using the uploaded file paths with proper session management
4. **Result Retrieval**: The resulting image is downloaded and converted back to ComfyUI tensor format

## Technical Details

- **API Integration**: Uses the official sm4ll-VTON Gradio API with proper SSE streaming
- **Session Management**: Maintains session state for reliable file uploads and downloads
- **Error Recovery**: Includes comprehensive error handling and fallback mechanisms
- **Format Support**: Handles various Gradio response formats and URL patterns

## Troubleshooting

- **Red Output Image**: Indicates an error occurred during processing. Check the console for detailed error messages.
- **Timeout Issues**: The API has a 5-minute timeout for processing. Large images or high server load may cause timeouts.
- **Upload Failures**: Check your internet connection and verify the Gradio space URL is accessible.

## Requirements

- ComfyUI
- torch
- torchvision  
- numpy
- Pillow
- requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- Built for the [sm4ll-VTON](https://huggingface.co/spaces/sm4ll-VTON/sm4ll-VTON-Demo) AI model
- Compatible with ComfyUI custom node architecture 
