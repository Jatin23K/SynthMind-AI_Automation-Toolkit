import pandas as pd
from data_purifier.agents.meta_analyzer_agent import MetaAnalyzerAgent

def test_meta_analyzer():
    # Initialize agent
    meta_analyzer = MetaAnalyzerAgent()
    
    # Test data
    test_dataset_paths = ["test_data.csv"]
    test_meta_output = "test_meta.txt"
    
    # Run analysis
    result, success = meta_analyzer.analyze(test_dataset_paths, test_meta_output)
    
    # Verify results
    assert success, "Meta analysis should succeed"
    assert "valid_dataset_paths" in result, "Result should contain valid dataset paths"
    assert "processing_instructions" in result, "Result should contain processing instructions"
    
    print("Meta Analyzer Test: PASSED")