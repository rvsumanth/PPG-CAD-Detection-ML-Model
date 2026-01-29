import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from .feature_engineering import HarmonizedPPGFeatureExtractor

class HarmonizedCADModel:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_extractor = HarmonizedPPGFeatureExtractor()
        self.optimal_threshold = 0.35
        self.is_trained = False

    def build_model(self):
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.5,
            class_weight='balanced',
            random_state=self.random_state,
            verbosity=-1
        )

        base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=8,
                min_samples_split=10, min_samples_leaf=5,
                class_weight='balanced_subsample',
                max_features=0.5, random_state=self.random_state,
                n_jobs=-1
            )),
            ('xgb', xgb.XGBClassifier(
                n_estimators=150, max_depth=4,
                learning_rate=0.05, subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.5,
                scale_pos_weight=2.0,
                eval_metric='logloss',
                random_state=self.random_state,
                n_jobs=-1
            )),
            ('lgb', self.lgb_model)
        ]


        meta = LogisticRegression(
            C=0.2, penalty='l1', solver='liblinear',
            class_weight='balanced', max_iter=3000
        )

        self.model = CalibratedClassifierCV(
            StackingClassifier(base_models, meta, cv=5, n_jobs=-1),
            method='sigmoid',
            cv=5
        )

    def fit(self, X, y):
        # 1. Feature engineering + selection
        X_feat = self.feature_extractor.fit_transform(X, y)
        self.X_feat_ = X_feat  # numeric matrix actually used

        # 2. Store SELECTED feature names (🔥 KEY FIX)
        self.feature_names_ = self.feature_extractor.selected_feature_names_

        # 3. Scaling
        X_scaled = self.scaler.fit_transform(X_feat)

        # 4. Build and train models
        self.build_model()
        self.model.fit(X_scaled, y)

        # 5. Train LightGBM separately for feature importance
        self.lgb_model.fit(X_scaled, y)

        self.is_trained = True


    def predict_proba(self, X):
        X_feat = self.feature_extractor.transform(X)
        X_scaled = self.scaler.transform(X_feat)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.optimal_threshold).astype(int)

    def save(self, path):
        joblib.dump(self, path)


    def get_feature_importance_model(self):
        self.feature_names_ = self.feature_extractor.selected_feature_names_
        return self.lgb_model

