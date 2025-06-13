#!/usr/bin/env python3
"""
Test a single model to debug TOV integration
"""

from eos_counsell_processor import CounsellEoSProcessor

def test_single_model():
    processor = CounsellEoSProcessor()
    
    # Test one specific model
    model_name = "DD2MEV_p80_CSS1_0.7_ncrit_0.192606_ecrit_185.223_pcrit_10.3131"
    
    print(f"Testing model: {model_name}")
    print("="*60)
    
    try:
        results = processor.process_full_analysis(model_name, stellar_mass=1.4)
        if results:
            print("\n✓ Analysis completed successfully!")
            print(f"Final stellar structure:")
            print(f"  R = {results['stellar_structure']['radius']:.2f} km")
            print(f"  M = {results['stellar_structure']['mass']:.3f} km")
            print(f"  f = {results['plot_coordinates']['frequency']:.1f} Hz")
            print(f"  |ΔΦ| = {results['plot_coordinates']['phase_shift']:.3e}")
        else:
            print("\n✗ Analysis failed")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    test_single_model()