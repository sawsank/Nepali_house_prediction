from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
import json
import os
from datetime import datetime

def get_web_search_tool():
    return DuckDuckGoSearchRun()
