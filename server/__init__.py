"""
Server module for LLM Prompt Auto-Tuner
Exports the FastAPI app for deployment
"""

from server.app import app

__all__ = ["app"]
