"""MAVIS local web application: a browser UI on top of the frozen OCR + LLM pipeline.

Nothing in prototype/ or postprocess/ is modified: this package re-uses ocr_engine.PrototypeOCREngine, the frame
selectors, PostProcessingPipeline and llm_stage.DocumentAssistant exactly as they are.
"""
