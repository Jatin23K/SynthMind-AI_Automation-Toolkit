from data_purifier.agents.process_recorder_agent import ProcessRecorderAgent

def test_recorder():
    # Initialize agent
    recorder = ProcessRecorderAgent()
    
    # Test recording
    test_logs = {
        'cleaning': ['Removed duplicates', 'Handled missing values'],
        'modification': ['Created new features'],
        'transformation': ['Scaled numeric features']
    }
    
    report = recorder.record_process(
        analysis_logs=test_logs['cleaning'],
        cleaning_logs=test_logs['cleaning'],
        modification_logs=test_logs['modification'],
        transformation_logs=test_logs['transformation']
    )
    
    # Verify results
    assert isinstance(report, str), "Report should be a string"
    assert "# Data Processing Report" in report, "Report should be in markdown format"
    
    print("Process Recorder Test: PASSED")