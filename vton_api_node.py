import os
import torch
import numpy as np
from PIL import Image
import requests
import time
import json
import math
import io
import random

def tensor_to_pil(tensor: torch.Tensor, batch_index=0):
    """Converts a ComfyUI image tensor to a PIL Image (RGB)."""
    try:
        # Ensure tensor is on CPU and detached
        tensor = tensor.detach().cpu()
        
        print(f"DEBUG: Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        print(f"DEBUG: Tensor min: {tensor.min():.3f}, max: {tensor.max():.3f}")
        
        # Handle batch dimension
        if tensor.ndim == 4:  # BCHW format
            tensor = tensor[batch_index]
            print(f"DEBUG: After batch selection: {tensor.shape}")
        
        # Handle different tensor formats
        if tensor.ndim == 3:
            height, width, channels = tensor.shape
            
            # Standard ComfyUI format is HWC (Height, Width, Channels)
            if channels <= 4:  # RGB/RGBA
                image_np = tensor.numpy()
                print(f"DEBUG: Using HWC format: {image_np.shape}")
            else:
                # If channels > 4, assume it's CHW format
                if tensor.shape[0] <= 4:  # CHW
                    image_np = tensor.permute(1, 2, 0).numpy()
                    print(f"DEBUG: Converted CHW to HWC: {image_np.shape}")
                else:
                    # Fallback: assume HWC
                    image_np = tensor.numpy()
                    print(f"DEBUG: Fallback HWC: {image_np.shape}")
                    
        elif tensor.ndim == 2:  # Grayscale HW
            image_np = tensor.numpy()
            image_np = np.expand_dims(image_np, axis=2)  # Add channel dimension
            print(f"DEBUG: Grayscale expanded: {image_np.shape}")
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
        
        # Ensure proper range [0, 1] -> [0, 255]
        if image_np.max() <= 1.0:
            image_np = image_np * 255.0
        
        # Clip and convert to uint8
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        print(f"DEBUG: Final numpy shape: {image_np.shape}, dtype: {image_np.dtype}")
        
        # Handle different channel counts
        if image_np.shape[2] == 1:  # Grayscale
            return Image.fromarray(image_np.squeeze(2), 'L').convert('RGB')
        elif image_np.shape[2] == 3:  # RGB
            return Image.fromarray(image_np, 'RGB')
        elif image_np.shape[2] == 4:  # RGBA
            return Image.fromarray(image_np, 'RGBA').convert('RGB')
        else:
            # Use first 3 channels as RGB
            return Image.fromarray(image_np[:, :, :3], 'RGB')
            
    except Exception as e:
        print(f"ERROR in tensor_to_pil: {e}")
        print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        # Create a fallback image
        return Image.new('RGB', (512, 512), color=(255, 0, 0))

def pil_to_tensor(pil_image: Image.Image):
    """Converts a PIL Image to a ComfyUI image tensor (BHWC, float 0-1)."""
    try:
        # Ensure image is RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        print(f"DEBUG: PIL image size: {pil_image.size}, mode: {pil_image.mode}")
        
        # Convert to numpy array (HWC format)
        image_np = np.array(pil_image)
        print(f"DEBUG: Numpy array shape: {image_np.shape}, dtype: {image_np.dtype}")
        
        # Normalize to 0-1
        image_np = image_np.astype(np.float32) / 255.0
        
        # ComfyUI expects BHWC format (Batch, Height, Width, Channels)
        # Add batch dimension: HWC -> BHWC
        tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        print(f"DEBUG: Output tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        print(f"DEBUG: Tensor min: {tensor.min():.3f}, max: {tensor.max():.3f}")
        
        return tensor
        
    except Exception as e:
        print(f"ERROR in pil_to_tensor: {e}")
        # Create a simple fallback tensor in BHWC format
        fallback = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
        fallback[:, :, :, 0] = 1.0  # Make it red
        print(f"DEBUG: Created fallback tensor with shape: {fallback.shape}")
        return fallback

def mask_to_pil(mask_tensor: torch.Tensor):
    """Convert ComfyUI mask tensor to PIL B&W image."""
    try:
        # Ensure tensor is on CPU and detached
        mask_tensor = mask_tensor.detach().cpu()
        
        print(f"DEBUG: Input mask tensor shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}")
        print(f"DEBUG: Mask min: {mask_tensor.min():.3f}, max: {mask_tensor.max():.3f}")
        
        # Handle batch dimension if present
        if mask_tensor.ndim == 3:  # BMH format (Batch, Mask, Height) - remove batch
            mask_tensor = mask_tensor[0]
        elif mask_tensor.ndim == 4:  # BMHW format - remove batch and take first channel
            mask_tensor = mask_tensor[0, 0]
        elif mask_tensor.ndim == 2:  # HW format - already correct
            pass
        else:
            # Try to squeeze out single dimensions
            mask_tensor = mask_tensor.squeeze()
        
        # Convert to numpy
        mask_np = mask_tensor.numpy()
        print(f"DEBUG: Mask numpy shape: {mask_np.shape}")
        
        # Ensure 2D array (Height, Width)
        if mask_np.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape: {mask_np.shape}")
        
        # Convert to 0-255 range
        if mask_np.max() <= 1.0:
            mask_np = mask_np * 255.0
        
        # Convert to uint8
        mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
        
        # Create PIL image in grayscale mode (B&W)
        mask_pil = Image.fromarray(mask_np, 'L')
        print(f"DEBUG: Created mask PIL image: {mask_pil.size}, mode: {mask_pil.mode}")
        
        return mask_pil
        
    except Exception as e:
        print(f"ERROR in mask_to_pil: {e}")
        print(f"Mask tensor shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}")
        # Create a fallback white mask
        return Image.new('L', (512, 512), color=255)

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

def call_vton_api(base_file_path, product_file_path, model_choice, base_url, session, mask_file_path=None):
    """Call VTON API following the exact Gradio API pattern (like curl -N)."""
    
    # Map ComfyUI model choices to API parameters
    model_mapping = {
        "eyewear": "eyewear",
        "footwear": "footwear", 
        "full-body": "dress"  # API expects "dress" for full-body garments
    }
    
    api_model_choice = model_mapping.get(model_choice, model_choice)
    print(f"\n🎨 Calling VTON API with model: {model_choice} → {api_model_choice}")
    
    # Gradio API always expects 4 parameters: [base, product, model, mask]
    # Use user-provided mask if available, otherwise pass null for backend fallback
    if mask_file_path:
        mask_parameter = {"path": mask_file_path, "meta": {"_type": "gradio.FileData"}}
        print(f"  🎭 Including user-provided mask in API call: {mask_file_path}")
    else:
        mask_parameter = None
        print(f"  🎭 No mask provided - sending null (backend will use base image fallback with default workflow)")
    
    api_data = {
        "data": [
            {"path": base_file_path, "meta": {"_type": "gradio.FileData"}},
            {"path": product_file_path, "meta": {"_type": "gradio.FileData"}},
            api_model_choice,
            mask_parameter
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
                                
                                # Handle error events
                                if event_type == 'error':
                                    print(f"  ❌ API returned error event - this usually means:")
                                    print(f"     - Images are invalid format/size")
                                    print(f"     - Server is overloaded")
                                    print(f"     - Model choice is invalid")
                                    print(f"     - Images are too large/small for the model")
                                    return None
                        
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
                "base_person_mask": ("MASK",),  # Optional mask input (MASK type)
             }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_vton"
    CATEGORY = "sm4ll/VTON"
    
    # Disable caching - always execute even with same inputs
    NOT_IDEMPOTENT = True
    
    def process_vton(self, base_person_image, product_image, model_choice, base_person_mask=None):
        try:
            # Generate internal cache-buster to force re-execution
            cache_buster = time.time() + random.random()
            print(f"🎲 Internal cache-buster: {cache_buster:.6f} (ensures fresh execution)")
            
            # Add imperceptible noise to force different input hash (ComfyUI cache bypass)
            # This is tiny enough to not affect the visual result but changes the tensor hash
            noise_scale = 1e-6  # Extremely small noise
            noise = torch.randn_like(base_person_image) * noise_scale
            base_person_image_noisy = base_person_image + noise
            print(f"🔄 Added imperceptible noise (scale: {noise_scale}) to bypass ComfyUI caching")
            
            # Use the hardcoded Gradio space URL
            base_url = "https://sm4ll-vton-sm4ll-vton-demo.hf.space"
            
            # Debug tensor shapes
            print(f"Input base tensor shape: {base_person_image_noisy.shape}")
            print(f"Input product tensor shape: {product_image.shape}")
            
            # Convert tensors to PIL images (use the slightly modified base image)
            base_pil = tensor_to_pil(base_person_image_noisy)
            product_pil = tensor_to_pil(product_image)
            
            print(f"Converted base image size: {base_pil.size}")
            print(f"Converted product image size: {product_pil.size}")
            
            # Validate minimum size requirements
            if base_pil.size[0] < 100 or base_pil.size[1] < 100:
                raise Exception(f"Base image too small: {base_pil.size}. Minimum 100x100 required.")
            if product_pil.size[0] < 100 or product_pil.size[1] < 100:
                raise Exception(f"Product image too small: {product_pil.size}. Minimum 100x100 required.")
            
            # Resize images to 1.62mpx using Lanczos interpolation
            base_resized = resize_to_megapixels(base_pil, 1.62)
            product_resized = resize_to_megapixels(product_pil, 1.62)
            
            print(f"Resized base image size: {base_resized.size}")
            print(f"Resized product image size: {product_resized.size}")
            
            # Ensure images are RGB (sometimes they come as RGBA or other formats)
            if base_resized.mode != 'RGB':
                print(f"Converting base image from {base_resized.mode} to RGB")
                base_resized = base_resized.convert('RGB')
            if product_resized.mode != 'RGB':
                print(f"Converting product image from {product_resized.mode} to RGB")
                product_resized = product_resized.convert('RGB')
                
            # Validate aspect ratio (VTON models usually expect reasonable aspect ratios)  
            base_aspect = base_resized.size[0] / base_resized.size[1]
            product_aspect = product_resized.size[0] / product_resized.size[1]
            print(f"Base image aspect ratio: {base_aspect:.2f}")
            print(f"Product image aspect ratio: {product_aspect:.2f}")
            
            if base_aspect < 0.3 or base_aspect > 3.0:
                print(f"⚠️  Warning: Base image has extreme aspect ratio: {base_aspect:.2f}")
            if product_aspect < 0.3 or product_aspect > 3.0:
                print(f"⚠️  Warning: Product image has extreme aspect ratio: {product_aspect:.2f}")
            
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
            
            # Handle optional mask image
            mask_file_path = None
            if base_person_mask is not None:
                print("Processing and uploading mask image...")
                print(f"Input mask tensor shape: {base_person_mask.shape}")
                
                # Convert MASK tensor to B&W PIL image
                mask_pil = mask_to_pil(base_person_mask)
                print(f"Mask B&W image size: {mask_pil.size}, mode: {mask_pil.mode}")
                
                # Validate minimum size requirements for mask
                if mask_pil.size[0] < 100 or mask_pil.size[1] < 100:
                    print(f"⚠️  Warning: Mask image is very small ({mask_pil.size}), this might not work well")
                
                # Resize mask to same target as other images
                mask_resized = resize_to_megapixels(mask_pil, 1.62)
                print(f"Resized mask B&W image size: {mask_resized.size}")
                
                # Convert B&W mask to RGB for API upload (API expects IMAGE format)
                mask_resized_rgb = mask_resized.convert('RGB')
                print(f"Converted mask from {mask_resized.mode} to {mask_resized_rgb.mode} for API")
                
                # Upload mask as RGB image to Gradio
                mask_file_path = upload_to_gradio_session(mask_resized_rgb, base_url, session)
                
                if not mask_file_path:
                    raise Exception("Failed to upload mask image to Gradio space")
                
                print(f"Mask image uploaded: {mask_file_path}")
            else:
                print("No mask image provided - will use base image fallback")
            
            print(f"Base image uploaded: {base_file_path}")
            print(f"Product image uploaded: {product_file_path}")
            if mask_file_path:
                print(f"Mask image uploaded: {mask_file_path}")
            
            # Call the VTON API
            result_path_or_url = call_vton_api(base_file_path, product_file_path, model_choice, base_url, session, mask_file_path)
            
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
            placeholder = Image.new('RGB', (512, 512), color=(255, 0, 0))
            placeholder_tensor = pil_to_tensor(placeholder)
            print(f"Created error placeholder with shape: {placeholder_tensor.shape}")
            return (placeholder_tensor,)

NODE_CLASS_MAPPINGS = {
    "VTONAPINode": VTONAPINode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTONAPINode": "sm4ll Wrapper Sampler"
} 
