import streamlit as st
import pandas as pd
from pathlib import Path

from src.preprocessing import (
    load_data,
    clean_data,
    get_train_test_data,
    build_preprocessor
)

from src.model import train_xgboost
from src.shap_utils import (
    create_shap_explainer,
    explain_single_customer,
    get_top_features
)

from src.recommendation_engine import generate_precautions

print("Day 3")