from .test_meta_analyzer import test_meta_analyzer
from .test_cleaner import test_cleaner
from .test_modifier import test_modifier
from .test_transformer import test_transformer
from .test_recorder import test_recorder

def run_all_tests():
    print("Running all agent tests...")
    
    test_meta_analyzer()
    test_cleaner()
    test_modifier()
    test_transformer()
    test_recorder()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()