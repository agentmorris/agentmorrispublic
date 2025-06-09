"""

timm_inference.py

This module provides functionality for running inference with TIMM models,
both as a command-line tool and as an importable Python module.

Dan Morris (with help from AI), 2025

"""

#%% Imports

import os
import sys
import argparse
import json

import numpy as np

from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path

import torch
import torchvision.transforms as transforms

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import apply_test_time_pool
from PIL import Image
from tqdm import tqdm


#%% Support functions

def load_class_names(filename: str) -> List[str]:
    """
    Load class names from a file. Supports JSON and TXT formats.
    
    Args:
        filename: Path to the class names file (.json or .txt)
        
    Returns:
        List of class names
    """
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext == '.json':
        with open(filename, 'r') as f:
            class_names = json.load(f)
            # Handle different JSON formats (list or dict)
            if isinstance(class_names, dict):
                # If it's a dict, convert to a list using the keys (assumed to be indices)
                # Sort by numeric key if possible
                try:
                    max_idx = max(int(k) for k in class_names.keys())
                    result = [""] * (max_idx + 1)
                    for k, v in class_names.items():
                        result[int(k)] = v
                    return result
                except (ValueError, TypeError):
                    # If keys aren't numeric, just return values
                    return list(class_names.values())
            else:
                # If it's already a list, return it directly
                return class_names
    elif file_ext == '.txt':
        with open(filename, 'r') as f:
            # Each line is a class name
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported class names file format: {file_ext}. Use .json or .txt")


#%% Inference class

class TIMMInference:
    """
    Class for running inference with TIMM models.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        test_time_pool: bool = False,
        class_names: Optional[List[str]] = None,
        batch_size: int = 8,
        use_test_input_size_if_available: bool = True
    ):
        """
        Initialize the TIMM inference model.
        
        Args:
            model_name: Base model architecture name (e.g., 'resnet34', 'vit_base_patch16_224')
                If None and checkpoint_path is provided, architecture will be inferred from checkpoint.
            checkpoint_path: Path to your trained model weights
            device: Device to run inference on ('cuda', 'cpu', etc.)
            test_time_pool: Whether to use test time pooling
            class_names: List of class names corresponding to model outputs
            batch_size: Batch size for inference
            use_test_input_size_if_available: If 'test_input_size is available in the model use that 
                rather than the default uinput size.
        """

        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.test_time_pool = test_time_pool
        self.class_names = class_names
        self.batch_size = batch_size
        self.use_test_input_size_if_available = use_test_input_size_if_available
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self._load_model()
        self._setup_transforms()


    def _load_model(self):
        """
        Load the TIMM model.
        """

        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        # If checkpoint is provided but no model name, try to load from checkpoint directly
        if self.checkpoint_path and not self.model_name:
            print(f"Loading model from checkpoint {self.checkpoint_path}...")
            try:
                # Load checkpoint to get model info
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                
                num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]
                print('Loaded a checkpoint with {} classes'.format(num_classes))

                if self.class_names is not None:
                    assert num_classes == len(self.class_names), 'Class count mismatch'
                
                # Try to extract model name from checkpoint
                if 'model' in checkpoint:
                    # If checkpoint has full model object
                    self.model = checkpoint['model']
                    print(f"Loaded full model object from checkpoint")
                elif 'arch' in checkpoint or 'model_name' in checkpoint:
                    # If checkpoint has architecture name
                    arch_name = checkpoint.get('arch') or checkpoint.get('model_name')
                    self.model_name = arch_name
                    print(f"Detected model architecture: {self.model_name}")
                    self.model = timm.create_model(
                        self.model_name, 
                        pretrained=False,
                        num_classes=num_classes
                    )
                    # Load state dict
                    if 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Try loading the checkpoint directly as a state dict
                        self.model.load_state_dict(checkpoint)
                    print(f"Loaded state dictionary from checkpoint")
                else:
                    # Try to find model name in checkpoint keys
                    if isinstance(checkpoint, dict):
                        # Look for keys that might contain model information
                        possible_keys = ['model_name', 'arch', 'architecture', 'config']
                        for key in possible_keys:
                            if key in checkpoint and isinstance(checkpoint[key], str):
                                self.model_name = checkpoint[key]
                                break
                    
                    if not self.model_name:
                        raise ValueError("Could not automatically determine model architecture from checkpoint. "
                                        "Please specify model_name explicitly.")
                    
                    # Create model with detected architecture
                    self.model = timm.create_model(
                        self.model_name,
                        pretrained=False
                    )
                    
                    # Try loading state dict
                    try:
                        if 'state_dict' in checkpoint:
                            self.model.load_state_dict(checkpoint['state_dict'])
                        else:
                            self.model.load_state_dict(checkpoint)
                    except Exception as e:
                        raise ValueError(f"Error loading state dict: {e}")
                    
                    print(f"Created model with architecture {self.model_name} and loaded weights")
                    
            except Exception as e:
                raise ValueError(f"Failed to load model from checkpoint: {e}")
        else:
            # Original code path when model_name is provided
            print(f"Loading model {self.model_name}...")
            
            # Create model with or without checkpoint
            if self.checkpoint_path:
                self.model = timm.create_model(
                    self.model_name,
                    checkpoint_path=self.checkpoint_path
                )
            else:
                # Ensure model_name is provided when no checkpoint is given
                if not self.model_name:
                    raise ValueError("Either model_name or checkpoint_path must be provided")
                
                self.model = timm.create_model(
                    self.model_name,
                    pretrained=True
                )
        
        # Check if model has class labels defined in its config
        if hasattr(self.model, 'pretrained_cfg') and self.model.pretrained_cfg is not None:
            if 'label_names' in self.model.pretrained_cfg and not self.class_names:
                self.class_names = self.model.pretrained_cfg['label_names']
                print(f"Loaded {len(self.class_names)} class names from model configuration")
        
        # Apply test time pooling if requested
        if self.test_time_pool:
            config = resolve_data_config({}, model=self.model)
            self.model = apply_test_time_pool(self.model, config)
        
        # Set model to evaluation mode and move to device
        self.model.eval()

        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
                
        self.model = self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
        

    def _setup_transforms(self):
        """
        Setup image transformations based on model config.
        """

        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        
        if self.use_test_input_size_if_available:
             
            if hasattr(self.model, 'default_cfg') and ('test_input_size' in self.model.default_cfg):
                             
                cfg = self.model.default_cfg

                # This size is specified as [channels,width,heigth]; width and height are
                # always the same.
                input_size = cfg['test_input_size'][1]
                assert cfg['test_input_size'][1] == cfg['test_input_size'][2]

                print('Loaded input size {} from test_input_size'.format(input_size))
                
                self.transform = transforms.Compose([
                    transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        # These are ImageNet statistics; they are constants throughout the timm library
                        mean=cfg.get('mean', [0.485, 0.456, 0.406]),
                        std=cfg.get('std', [0.229, 0.224, 0.225])
                    )
                ])
            
            else:

                print('Could not find test input size')

        # ...if we should try to load the test-time inference size

        # Scrap code to messs around with custom transforms
        if False:
             self.transform = transforms.Compose([
                 # transforms.Resize((288, 288), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                 # transforms.CenterCrop(size=(288, 288)),
                 transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                 transforms.CenterCrop(size=(384, 384)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
             ])

        print(f"Created transforms: {self.transform}")
    

    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess a single image for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """

        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0)
        return tensor
    

    def infer_batch(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of preprocessed images.
        
        Args:
            image_tensors: Batch of image tensors
            
        Returns:
            Model output
        """

        image_tensors = image_tensors.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensors)
        
        return output
    

    def postprocess_output(self, output: torch.Tensor, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Process model output into human-readable format.
        
        Args:
            output: Model output tensor
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with prediction results
        """

        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
        
        results = []
        for i in range(output.shape[0]):  # For each item in batch
            item_results = {
                'class_indices': top_indices[i].cpu().numpy().tolist(),
                'probabilities': top_probs[i].cpu().numpy().tolist(),
            }
            
            if self.class_names:
                item_results['class_names'] = [self.class_names[idx] for idx in item_results['class_indices']]
            
            results.append(item_results)
        
        return results
    

    def infer_on_images(self, 
                        image_paths: List[Union[str, Path]],
                        top_k: int = 1,
                        report_progress = True) -> List[Dict[str, Any]]:
        """
        Run inference on a list of images.
        
        Args:
            image_paths: List of paths to image files
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with prediction results
        """

        results = []
        
        total_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size
    
        batch_range = range(0, len(image_paths), self.batch_size)
        
        if report_progress:
            batch_range = tqdm(batch_range, total=total_batches, desc="Processing images", unit="batch")
        
        # Process images in batches
        for i in batch_range:
            batch_paths = image_paths[i:i + self.batch_size]
            batch_tensors = torch.cat([self.preprocess_image(path) for path in batch_paths])
            
            output = self.infer_batch(batch_tensors)
            batch_results = self.postprocess_output(output, top_k)
            
            # Add image paths to results
            for j, path in enumerate(batch_paths):
                batch_results[j]['image_path'] = str(path)
            
            results.extend(batch_results)
        return results
    

    def infer_on_folder(self, 
                        folder_path: Union[str, Path], 
                        top_k: int = 1,
                        recursive: bool = False) -> List[Dict[str, Any]]:
        """
        Run inference on all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            top_k: Number of top predictions to return
            recursive: Whether to search recursively in subdirectories
            
        Returns:
            List of dictionaries with prediction results
        """

        folder_path = Path(folder_path)
        image_paths = []
        
        # Find all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        if recursive:
            # Recursive search through all subdirectories
            for ext in valid_extensions:
                image_paths.extend(folder_path.glob(f'**/*{ext}'))
                image_paths.extend(folder_path.glob(f'**/*{ext.upper()}'))
        else:
            # Non-recursive search (original behavior)
            for file_path in folder_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    image_paths.append(file_path)
        
        print(f"Found {len(image_paths)} images in {folder_path}{' (including subdirectories)' if recursive else ''}")
        
        if not image_paths:
            print("No images found in the specified folder.")
            return []
        
        return self.infer_on_images(image_paths, top_k)


    def convert_checkpoint_for_inference(checkpoint_path, output_path=None):
        """
        Convert a training checkpoint to an inference-ready checkpoint.
        Only keeps the model weights, discarding optimizer state and other training data.
        """
        
        if output_path is None:
            # Create a new filename with '_inference' suffix
            output_path = checkpoint_path.replace('.pth', '_inference.pth')
            if output_path == checkpoint_path:
                output_path = checkpoint_path + '.inference'
        
        # Load the original checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create a new checkpoint with only the necessary components
        inference_checkpoint = {
            'state_dict': checkpoint['state_dict'],
            'arch': checkpoint.get('arch', None),             
        }
        
        # Save the new checkpoint
        torch.save(inference_checkpoint, output_path)
        print(f"Created inference checkpoint at {output_path}")
        return output_path


#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser(description='Run inference with TIMM models.')
    
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to your trained model weights')
    parser.add_argument('--folder', type=str, required=True,
                        help='Folder containing images for inference')
    parser.add_argument('--device', type=str,
                        help='Device to run inference on (e.g., cuda, cpu)')
    parser.add_argument('--test_time_pool', action='store_true',
                        help='Use test time pooling')
    parser.add_argument('--class_names_file', type=str,
                        help='File containing class names (.json or .txt format)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to return')
    parser.add_argument('--output_file', type=str,
                        help='Output file to save results (JSON)')
    parser.add_argument('--model_name', type=str,
                        help='Base model architecture name (optional if checkpoint provided)')
    parser.add_argument('--recursive', action='store_true',
                        help='Search for images recursively in subdirectories')
    parser.add_argument('--no_use_test_image_size', action='store_true',
                        help="Don't use test_image_size, even if it's specified in the model")
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    
    # Validate that either model_name or checkpoint_path is provided
    if not args.model_name and not args.checkpoint_path:
        print("Error: Either --model_name or --checkpoint_path must be provided")
        parser.print_help()
        parser.exit(1)

    # Load class names if provided
    class_names = None
    if args.class_names_file:
        try:
            class_names = load_class_names(args.class_names_file)
            print(f"Loaded {len(class_names)} class names from {args.class_names_file}")
        except Exception as e:
            print(f"Error loading class names: {e}")
            sys.exit(1)
    
    # Initialize inference
    inference = TIMMInference(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        test_time_pool=args.test_time_pool,
        class_names=class_names,
        batch_size=args.batch_size,
        use_test_input_size_if_available=(not args.no_use_test_image_size)
    )
    
    # Run inference
    results = inference.infer_on_folder(args.folder, top_k=args.top_k, recursive=args.recursive)

    
    # Print or save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")
    else:
        for result in results:
            print(f"\nImage: {result['image_path']}")
            for i in range(len(result['class_indices'])):
                class_idx = result['class_indices'][i]
                prob = result['probabilities'][i]
                
                if 'class_names' in result:
                    class_name = result['class_names'][i]
                    print(f"  {class_name} ({class_idx}): {prob:.4f}")
                else:
                    print(f"  Class {class_idx}: {prob:.4f}")


if __name__ == '__main__':
    main()
