import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Setup path to import config dynamically
# __file__ looks at src/features/engineer.py, so parent.parent gets us exactly to 'src/'
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_DIR))
import config

# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure features directory exists
FEATURES_DIR = config.PROJECT_ROOT / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def extract_and_join_data():
    """Extracts data from the Star Schema into a single flat analysis DataFrame."""
    logger.info("Extracting and joining data from warehouse...")
    engine = create_engine(config.DATABASE_URI)
    
    # SQL Query joining Fact with all Dimensions
    query = """
    SELECT 
        f.fact_id, f.gpa, f.absences, f.risk_score, f.risk_label, f.checkpoint_week,
        s.age, s.sex, s.address, s.family_size, s.parent_edu, s.internet_access, s.paid_classes,
        c.subject, c.study_time, c.failures_history, c.extracurricular,
        d.nationality, d.socioeconomic_tier, d.first_gen_student, d.scholarship_flag
    FROM fact_student_risk f
    JOIN dim_student s ON f.student_id = s.student_id
    JOIN dim_course c ON f.course_id = c.course_id
    JOIN dim_demographics d ON f.demo_id = d.demo_id
    """
    
    df = pd.read_sql(query, engine)
    
    # Convert categorical text columns to numeric using One-Hot Encoding for the correlation matrix/models
    # (Assuming columns like 'sex', 'address', 'subject' are still text strings)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Save raw flat file
    df.to_csv(FEATURES_DIR / "analysis_df.csv", index=False)
    logger.info(f"Analysis dataframe exported. Shape: {df.shape}")
    return df

def generate_correlation_heatmap(df):
    """Generates and saves the Pearson correlation matrix heatmap."""
    logger.info("Generating feature correlation heatmap...")
    plt.figure(figsize=(16, 12))
    # Drop IDs and labels for the correlation matrix
    cols_to_drop = ['fact_id', 'risk_label', 'checkpoint_week']
    df_corr = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).corr()
    
    sns.heatmap(df_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(config.PROJECT_ROOT / "feature_correlation_heatmap.png")
    plt.close()
    logger.info("Saved feature_correlation_heatmap.png")
    return df_corr

def remove_collinear_features(df, threshold=0.85):
    """Removes one of two features that have a correlation higher than the threshold."""
    logger.info(f"Removing multicollinear features (r > {threshold})...")
    # Exclude target and ID from correlation logic
    features = df.drop(columns=['fact_id', 'risk_label', 'checkpoint_week'], errors='ignore')
    corr_matrix = features.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    df_reduced = df.drop(columns=to_drop)
    logger.info(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")
    return df_reduced

def process_checkpoint(df, checkpoint_week, name):
    """Filters data for a specific checkpoint, applies selection, splitting, SMOTE, and scaling."""
    logger.info(f"--- Processing {name} ---")
    
    # Filter by checkpoint week
    df_cp = df[df['checkpoint_week'] == checkpoint_week].copy()
    
    # --- THE FIX: Drop any rows that don't have a target label ---
    df_cp = df_cp.dropna(subset=['risk_label'])
    
    if df_cp.empty:
         logger.warning(f"No labeled data found for {name}. Skipping.")
         return
         
    # Drop ID and checkpoint columns for X
    X = df_cp.drop(columns=['fact_id', 'risk_label', 'checkpoint_week'], errors='ignore')
    y = df_cp['risk_label']
    
    # Fill any remaining NaNs in features (X) with 0
    X.fillna(0, inplace=True)
    
    # 1. Feature Selection (SelectKBest)
    logger.info("Selecting top 15 features...")
    k = min(15, X.shape[1]) 
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()]
    X = pd.DataFrame(X_selected, columns=selected_features)
    
    # 2. Train / Val / Test Split (70/15/15)
    # First split: 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    # Second split: Split the 30% Temp into 15% Val and 15% Test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    # 3. Class Imbalance - SMOTE (Applied ONLY to Train!)
    logger.info(f"Class ratio before SMOTE: 0: {sum(y_train==0)}, 1: {sum(y_train==1)}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Class ratio AFTER SMOTE: 0: {sum(y_train_resampled==0)}, 1: {sum(y_train_resampled==1)}")
    
    # 4. Feature Scaling (Fit on Train ONLY!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference later
    joblib.dump(scaler, config.PROJECT_ROOT / f"scaler_{name}.joblib")
    
    # Save final matrices to disk for modeling
    # We combine X and y back together just for easy storage in CSV
    train_out = pd.DataFrame(X_train_scaled, columns=selected_features)
    train_out['target'] = y_train_resampled.values
    
    val_out = pd.DataFrame(X_val_scaled, columns=selected_features)
    val_out['target'] = y_val.values
    
    test_out = pd.DataFrame(X_test_scaled, columns=selected_features)
    test_out['target'] = y_test.values
    
    # Save to the features folder
    train_out.to_csv(FEATURES_DIR / f"{name}_train.csv", index=False)
    val_out.to_csv(FEATURES_DIR / f"{name}_val.csv", index=False)
    test_out.to_csv(FEATURES_DIR / f"{name}_test.csv", index=False)
    logger.info(f"Successfully engineered and saved {name} datasets.\n")

if __name__ == "__main__":
    # 1. Extract and Join
    flat_df = extract_and_join_data()
    
    # 2. Correlation Analysis
    generate_correlation_heatmap(flat_df)
    reduced_df = remove_collinear_features(flat_df)
    
    # 3. Process each checkpoint independently
    # Assuming checkpoint_week contains numeric values like 4, 8, 12. 
    # Adjust the integer values below if your DB uses string labels like 'Week 4'.
    process_checkpoint(reduced_df, checkpoint_week=4, name="checkpoint_4")
    process_checkpoint(reduced_df, checkpoint_week=8, name="checkpoint_8")
    process_checkpoint(reduced_df, checkpoint_week=12, name="checkpoint_12")
    
    logger.info("Phase 4 Feature Engineering complete!")