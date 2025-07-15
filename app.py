"""
🤖 RAG-Powered Data Analysis Application
An intelligent Streamlit app that combines Retrieval-Augmented Generation (RAG) 
with advanced data science techniques for interactive CSV data analysis.

Features:
- CSV file upload and processing
- Natural language querying with RAG
- AI-powered data insights and analysis
- Dynamic visualizations with Plotly
- Statistical analysis and recommendations

Author: AIML-alchemist
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
import os
from datetime import datetime
