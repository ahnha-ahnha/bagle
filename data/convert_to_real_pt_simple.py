#!/usr/bin/env python3
"""
Convert ADNI pickle files to real.pt format for bagle project
"""

import pickle
import torch
import os

def convert_pickle_to_real_pt(pickle_path, output_path):
    """
    Convert pickle file to real.pt format
    
    Args:
        pickle_path: Path to the pickle file
        output_path: Path to save the real.pt file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load pickle file
        print(f"Loading pickle file from: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        print("Keys in pickle file:", data.keys() if isinstance(data, dict) else "Not a dict")
        
        # Extract the main feature data (usually 'X' contains the real data)
        if isinstance(data, dict):
            if 'X' in data:
                real_data = data['X']
                print(f"Extracted X with shape: {real_data.shape}")
                print(f"Data type: {type(real_data)}")
                
                # Convert to tensor if it's not already
                if not isinstance(real_data, torch.Tensor):
                    real_data = torch.tensor(real_data)
                
                # Save as real.pt
                torch.save(real_data, output_path)
                print(f"✓ Successfully saved real.pt to: {output_path}")
                return True
            else:
                print("✗ No 'X' key found in data")
                return False
        else:
            print("✗ Data is not a dictionary")
            return False
            
    except Exception as e:
        print(f"✗ Error converting {pickle_path}: {e}")
        return False

def main():
    # Define the mapping of pickle files to output paths
    conversions = [
        ('/home/user14/AGT/data/pickle/adni_CT_0.pickle', '/home/user14/bagle/data/ADNI_CT/real.pt'),
        ('/home/user14/AGT/data/pickle/adni_Amy_0.pickle', '/home/user14/bagle/data/ADNI_Amy/real.pt'),
        ('/home/user14/AGT/data/pickle/adni_FDG_0.pickle', '/home/user14/bagle/data/ADNI_FDG/real.pt'),
        ('/home/user14/AGT/data/pickle/adni_Tau_0.pickle', '/home/user14/bagle/data/ADNI_Tau/real.pt'),
    ]
    
    successful_conversions = []
    failed_conversions = []
    
    for pickle_path, output_path in conversions:
        if os.path.exists(pickle_path):
            print(f"\n{'='*60}")
            print(f"Converting {pickle_path}")
            print(f"To: {output_path}")
            print(f"{'='*60}")
            
            success = convert_pickle_to_real_pt(pickle_path, output_path)
            if success:
                successful_conversions.append((pickle_path, output_path))
            else:
                failed_conversions.append((pickle_path, output_path))
        else:
            print(f"Warning: {pickle_path} not found!")
            failed_conversions.append((pickle_path, output_path))
    
    # Print summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    
    if successful_conversions:
        print(f"✓ Successfully converted {len(successful_conversions)} files:")
        for pickle_path, output_path in successful_conversions:
            print(f"  • {os.path.basename(pickle_path)} → {output_path}")
    
    if failed_conversions:
        print(f"\n✗ Failed to convert {len(failed_conversions)} files:")
        for pickle_path, output_path in failed_conversions:
            print(f"  • {os.path.basename(pickle_path)} → {output_path}")
        
        print(f"\nNote: The failed files appear to be corrupted or incomplete.")
        print(f"You may need to regenerate these pickle files from the original data.")
    
    print(f"\nConversion completed!")

if __name__ == "__main__":
    main()
