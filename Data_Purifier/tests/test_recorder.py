from ..agents.process_recorder_agent import ProcessRecorderAgent


def test_recorder():
    # Initialize agent
    recorder = ProcessRecorderAgent()

    # Simulate recording activities
    recorder.record_task_activity("TestAgent", "Performing analysis", "Success", {"details": "Analysis completed"})
    recorder.add_stage_report("cleaning", {"logs": ["Removed duplicates", "Handled missing values"]})
    recorder.add_stage_report("modification", {"logs": ["Created new features"]})
    recorder.add_stage_report("transformation", {"logs": ["Scaled numeric features"]})

    # Generate final report
    report = recorder.generate_final_report(output_path="test_report")

    # Verify results
    assert isinstance(report, dict), "Report should be a dictionary"
    assert "pipeline_activities" in report, "Report should contain pipeline activities"
    assert "stage_summaries" in report, "Report should contain stage summaries"
    assert len(report["pipeline_activities"]) > 0, "Should have recorded activities"
    assert "cleaning" in report["stage_summaries"], "Should have cleaning stage summary"

    print("Process Recorder Test: PASSED")
