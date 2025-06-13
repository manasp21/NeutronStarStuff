#!/usr/bin/env python3
"""
Test script to verify phase transition detection for specific model
"""

from eos_counsell_processor import CounsellEoSProcessor

def test_phase_detection():
    processor = CounsellEoSProcessor()
    
    # Test the specific model
    model_name = "DD2MEV_p80_CSS1_0.7_ncrit_0.192606_ecrit_185.223_pcrit_10.3131"
    
    print(f"Testing phase transition detection for: {model_name}")
    print("="*60)
    
    # Load data
    data = processor.load_eos_data(model_name)
    print(f"Loaded {len(data)} data points")
    
    # Test phase transition detection
    transition = processor.step1_identify_phase_transition(data)
    
    if transition['relative_jump'] > 0:
        print("\n✓ Phase transition successfully detected!")
        print(f"Found transition with {transition['relative_jump']:.1%} energy density jump")
    else:
        print("\n✗ No phase transition detected")
        
        # Manual check around expected transition point
        print("\nManual verification around ε ≈ 185 MeV/fm³:")
        for point in data:
            if 180 <= point['epsilon'] <= 190:
                print(f"  ε = {point['epsilon']:.3f}, p = {point['p']:.6f}")

if __name__ == "__main__":
    test_phase_detection()