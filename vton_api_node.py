import os
import torch
import numpy as np
from PIL import Image
import requests
import time
import json
import math
import io

def tensor_to_pil(tensor: torch.Tensor, batch_index=0):
    """Converts a ComfyUI image tensor (BCHW, float 0-1) to a PIL Image (RGB)."""
    if tensor.ndim == 4:
        tensor = tensor[batch_index]
    # Permute CHW to HWC, scale, convert to numpy, then to PIL
    image_np = tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    if image_np.shape[2] == 1:  # Grayscale tensor
        return Image.fromarray(image_np.squeeze(), 'L').convert('RGB')
    return Image.fromarray(image_np, 'RGB')

def pil_to_tensor(pil_image: Image.Image):
    """Converts a PIL Image to a ComfyUI image tensor (BCHW, float 0-1)."""
    # Convert to numpy array
    image_np = np.array(pil_image)
    # Normalize to 0-1
    image_np = image_np.astype(np.float32) / 255.0
    # Convert to tensor and rearrange dimensions
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW
    return tensor

def resize_to_megapixels(image: Image.Image, target_mpx: float = 1.62):
    """Resize image to target megapixels using Lanczos interpolation."""
    current_pixels = image.width * image.height
    target_pixels = target_mpx * 1_000_000
    
    if current_pixels <= target_pixels:
        return image  # No need to resize if already smaller
    
    scale_factor = math.sqrt(target_pixels / current_pixels)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def upload_to_gradio_session(image, base_url, session):
    """Upload image using a session for state persistence."""
    try:
        # Convert image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Try the standard Gradio upload endpoint
        upload_url = f"{base_url}/gradio_api/upload"
        
        # Prepare the file for upload
        files = {
            'files': ('image.png', img_buffer, 'image/png')
        }
        
        print(f"Uploading to Gradio space with session: {upload_url}")
        response = session.post(upload_url, files=files, timeout=30)
        
        if response.status_code == 200:
            # Parse the response to get the file path/URL
            try:
                result = response.json()
                print(f"Session upload response: {result}")  # Debug: show full response
                
                # Different Gradio versions return different formats
                if isinstance(result, list) and len(result) > 0:
                    file_info = result[0]
                    print(f"Session file info: {file_info}")  # Debug: show file info
                    
                    if isinstance(file_info, dict):
                        # Format: [{"name": "filename", "data": "file_path", ...}]
                        file_path = file_info.get('name') or file_info.get('data') or file_info.get('path')
                        if file_path:
                            full_url = f"{base_url}/file={file_path}"
                            print(f"Session upload successful: {full_url}")
                            return file_path  # Return the internal file path for API calls
                    elif isinstance(file_info, str):
                        # Format: ["/tmp/gradio/hash/filename"]
                        file_path = file_info
                        full_url = f"{base_url}/file={file_path}"
                        print(f"Session upload successful: {full_url}")
                        return file_path  # Return the internal file path for API calls
                elif isinstance(result, dict):
                    # Some formats return a dict directly
                    file_path = result.get('name') or result.get('data') or result.get('path')
                    if file_path:
                        full_url = f"{base_url}/file={file_path}"
                        print(f"Session upload successful: {full_url}")
                        return file_path  # Return the internal file path for API calls
                elif isinstance(result, str):
                    # Try to use the raw response as filename
                    file_path = result
                    full_url = f"{base_url}/file={file_path}"
                    print(f"Session upload successful: {full_url}")
                    return file_path  # Return the internal file path for API calls
                        
                print(f"Unexpected session upload response format: {result}")
                
            except json.JSONDecodeError:
                # Sometimes the response is just a filename string
                print(f"Session JSON decode failed, raw response: '{response.text}'")  # Debug
                file_path = response.text.strip().strip('"')
                if file_path:
                    full_url = f"{base_url}/file={file_path}"
                    print(f"Session upload successful: {full_url}")
                    return file_path  # Return the internal file path for API calls
        else:
            print(f"Session upload failed with status {response.status_code}: {response.text}")
        
    except Exception as e:
        print(f"Error in session upload to Gradio space: {e}")
    
    return None

def call_vton_api(base_file_path, product_file_path, model_choice, base_url, session):
    """Call VTON API following the exact Gradio API pattern (like curl -N)."""
    print(f"\n🎨 Calling VTON API with model: {model_choice}")
    
    # Prepare the API call data exactly as shown in the YAML
    api_data = {
        "data": [
            {"path": base_file_path, "meta": {"_type": "gradio.FileData"}},
            {"path": product_file_path, "meta": {"_type": "gradio.FileData"}},
            model_choice
        ]
    }
    
    print(f"  📤 API request data: {api_data}")
    
    try:
        # Step 1: POST to get EVENT_ID (exactly like the YAML example)
        submit_url = f"{base_url}/gradio_api/call/generate"
        print(f"  🚀 Submitting job to: {submit_url}")
        
        response = session.post(
            submit_url,
            json=api_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"  📨 Submit response: {response.text}")
        
        if response.status_code != 200:
            print(f"  ❌ API submit failed: {response.status_code}")
            return None
        
        # Extract EVENT_ID (like awk -F'"' '{ print $4}' in the YAML)
        try:
            event_data = response.json()
            event_id = event_data.get('event_id')
            if not event_id:
                print(f"  ❌ No event_id in response: {event_data}")
                return None
        except:
            print(f"  ❌ Failed to parse event_id from response")
            return None
            
        print(f"  ✅ Got EVENT_ID: {event_id}")
        
        # Step 2: GET with streaming (equivalent to curl -N)
        stream_url = f"{base_url}/gradio_api/call/generate/{event_id}"
        print(f"  🌊 Starting SSE stream: {stream_url}")
        print(f"     (equivalent to: curl -N {stream_url})")
        
        # Make streaming request exactly like curl -N
        stream_response = session.get(
            stream_url,
            headers={
                'Accept': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            },
            timeout=300,  # 5 minutes for AI processing
            stream=True
        )
        
        if stream_response.status_code != 200:
            print(f"  ❌ Stream failed: {stream_response.status_code}")
            print(f"  📄 Response: {stream_response.text[:200]}")
            return None
        
        print(f"  ✅ SSE stream connected (status: {stream_response.status_code})")
        
        # Process streaming response line by line (like curl -N output)
        buffer = ""
        start_time = time.time()
        
        for chunk in stream_response.iter_content(chunk_size=1, decode_unicode=True):
            if chunk:
                buffer += chunk
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line:
                        elapsed = time.time() - start_time
                        print(f"  📡 [{elapsed:.1f}s] {line}")
                        
                        # Handle SSE data lines
                        if line.startswith('data: '):
                            data_content = line[6:]  # Remove 'data: ' prefix
                            
                            # Skip empty data
                            if not data_content or data_content == '{}':
                                continue
                            
                            try:
                                # Parse JSON data
                                if data_content.startswith('{') or data_content.startswith('['):
                                    result_data = json.loads(data_content)
                                    
                                    if isinstance(result_data, dict):
                                        # Check for completion
                                        if result_data.get('msg') == 'process_completed':
                                            output = result_data.get('output', {})
                                            if output and 'data' in output and output['data']:
                                                result_path = output['data'][0]
                                                print(f"  ✅ COMPLETED! Result: {result_path}")
                                                return result_path
                                        
                                        # Check for failure
                                        elif result_data.get('msg') == 'process_failed':
                                            print(f"  ❌ FAILED: {result_data}")
                                            return None
                                        
                                        # Status updates
                                        elif result_data.get('msg') in ['process_starts', 'estimation']:
                                            print(f"  ⏳ Status: {result_data.get('msg')}")
                                        
                                        # Progress updates
                                        elif 'progress' in result_data:
                                            progress = result_data.get('progress', '')
                                            print(f"  🔄 Progress: {progress}")
                                    
                                    elif isinstance(result_data, list) and result_data:
                                        # Handle Gradio FileData objects in array (THIS IS THE WORKING FORMAT!)
                                        first_item = result_data[0]
                                        
                                        # Check if it's a FileData object with path/url
                                        if isinstance(first_item, dict):
                                            if 'url' in first_item:
                                                result_url = first_item['url']
                                                print(f"  ✅ COMPLETED! Result URL: {result_url}")
                                                return result_url
                                            elif 'path' in first_item:
                                                result_path = first_item['path']
                                                print(f"  ✅ COMPLETED! Result path: {result_path}")
                                                return result_path
                                        
                                        # Handle string paths
                                        elif isinstance(first_item, str) and first_item.startswith('/'):
                                            print(f"  ✅ COMPLETED! Result: {first_item}")
                                            return first_item
                                
                                # Handle direct string responses (file paths)
                                elif data_content.startswith('/') or data_content.startswith('"/"'):
                                    result_path = data_content.strip('"')
                                    print(f"  ✅ COMPLETED! Result: {result_path}")
                                    return result_path
                                
                            except json.JSONDecodeError:
                                # Try to extract file path from non-JSON data
                                if data_content.startswith('/'):
                                    print(f"  ✅ COMPLETED! Result: {data_content}")
                                    return data_content
                                else:
                                    print(f"  📝 Raw data: {data_content}")
                        
                        # Handle other SSE lines
                        elif line.startswith('event: '):
                            event_type = line[7:]
                            if event_type != 'heartbeat':  # Don't log heartbeats
                                print(f"  🎯 Event: {event_type}")
                        
                        # Connection heartbeat
                        elif line.startswith('id: ') or line.startswith('retry: '):
                            continue  # Skip SSE metadata
                
                # Timeout check
                if time.time() - start_time > 300:  # 5 minutes
                    print(f"  ⏰ Stream timeout after 5 minutes")
                    break
        
        print(f"  🔚 Stream ended without result")
        return None
        
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return None

def download_result_image(result_path_or_url, base_url, session):
    """Download the result image from Gradio."""
    if not result_path_or_url:
        return None
        
    print(f"\n📥 Downloading result image: {result_path_or_url}")
    
    # Check if it's already a complete URL
    if result_path_or_url.startswith('http'):
        possible_urls = [result_path_or_url]
    else:
        # Try different URL formats for the result path
        possible_urls = [
            f"{base_url}/gradio_api/file={result_path_or_url}",  # Most likely format for Gradio
            f"{base_url}/file={result_path_or_url}",
            f"{base_url}/file/{result_path_or_url}",
            f"{base_url}/files/{result_path_or_url}",
            f"{base_url}/api/file/{result_path_or_url}",
        ]
    
    for url in possible_urls:
        print(f"  🔗 Trying: {url}")
        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                if 'image' in content_type or response.content.startswith(b'\x89PNG') or response.content.startswith(b'\xFF\xD8\xFF'):
                    print(f"  ✅ Successfully downloaded image ({len(response.content)} bytes)")
                    
                    # Return PIL Image
                    image = Image.open(io.BytesIO(response.content))
                    print(f"  🖼️  Result image size: {image.size}")
                    return image
                else:
                    print(f"    ❌ Not an image: {content_type}")
            else:
                print(f"    ❌ HTTP {response.status_code}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    print("  ❌ Could not download result image")
    return None

class VTONAPINode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_person_image": ("IMAGE",),
                "product_image": ("IMAGE",),
                "model_choice": (["eyewear", "footwear", "full-body"], {"default": "eyewear"}),
            },
            "optional": {
                "gradio_space_url": ("STRING", {"default": "https://sm4ll-vton-sm4ll-vton-demo.hf.space", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_vton"
    CATEGORY = "sm4ll/VTON"
    
    def process_vton(self, base_person_image, product_image, model_choice, gradio_space_url="https://sm4ll-vton-sm4ll-vton-demo.hf.space"):
        try:
            # Clean up the URL
            base_url = gradio_space_url.strip().rstrip('/')
            
            # Convert tensors to PIL images
            base_pil = tensor_to_pil(base_person_image)
            product_pil = tensor_to_pil(product_image)
            
            print(f"Original base image size: {base_pil.size}")
            print(f"Original product image size: {product_pil.size}")
            
            # Resize images to 1.62mpx using Lanczos interpolation
            base_resized = resize_to_megapixels(base_pil, 1.62)
            product_resized = resize_to_megapixels(product_pil, 1.62)
            
            print(f"Resized base image size: {base_resized.size}")
            print(f"Resized product image size: {product_resized.size}")
            
            # Create a session to maintain cookies/state
            session = requests.Session()
            
            # Upload images directly to the Gradio space
            print("Uploading base image to Gradio space...")
            base_file_path = upload_to_gradio_session(base_resized, base_url, session)
            
            if not base_file_path:
                raise Exception("Failed to upload base image to Gradio space")
            
            print("Uploading product image to Gradio space...")
            product_file_path = upload_to_gradio_session(product_resized, base_url, session)
            
            if not product_file_path:
                raise Exception("Failed to upload product image to Gradio space")
            
            print(f"Base image uploaded: {base_file_path}")
            print(f"Product image uploaded: {product_file_path}")
            
            # Call the VTON API
            result_path_or_url = call_vton_api(base_file_path, product_file_path, model_choice, base_url, session)
            
            if not result_path_or_url:
                raise Exception("VTON API call failed - no result returned")
            
            # Download the result image
            result_image = download_result_image(result_path_or_url, base_url, session)
            
            if not result_image:
                raise Exception("Failed to download result image")
            
            # Convert result image back to tensor
            result_tensor = pil_to_tensor(result_image)
            print("✓ VTON processing completed successfully!")
            return (result_tensor,)
            
        except Exception as e:
            print(f"Error in VTON API processing: {e}")
            # Return a red placeholder image in case of error
            placeholder = Image.new('RGB', (512, 512), color='red')
            return (pil_to_tensor(placeholder),)

NODE_CLASS_MAPPINGS = {
    "VTONAPINode": VTONAPINode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTONAPINode": "VTON API Node"
} 