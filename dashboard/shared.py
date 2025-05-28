import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.font_manager as fm
import matplotlib as mpl

app_dir = Path(__file__).parent
df = pd.read_csv(app_dir / "./data/df_final.csv",index_col=0)

font_path = app_dir / "MaruBuri-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)

# 마이너스 깨짐 방지
mpl.rcParams["axes.unicode_minus"] = False
