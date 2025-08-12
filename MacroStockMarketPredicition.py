import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# FRED API for economic data
from fredapi import Fred

class StockMarketPredictor:
    def __init__(self, fred_api_key, stock_symbol='SPY', start_date='2010-01-01'):

        self.fred_api_key = fred_api_key
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.models = {}
        self.scalers = {}
        
    def collect_stock_data(self):
        """Collect stock data using yfinance"""
        print(f"Collecting stock data for {self.stock_symbol}...")
        stock = yf.Ticker(self.stock_symbol)
        stock_data = stock.history(start=self.start_date, end=self.end_date)
        
        # Remove timezone information to match FRED data
        stock_data.index = stock_data.index.tz_localize(None)
        
        # Calculate the technical indicators
        stock_data['Returns'] = stock_data['Close'].pct_change()
        stock_data['Log_Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        stock_data['Volatility'] = stock_data['Returns'].rolling(window=21).std()
        stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['RSI'] = self.calculate_rsi(stock_data['Close'])
        
        # Create target variables
        stock_data['Next_Day_Return'] = stock_data['Returns'].shift(-1)
        stock_data['Direction'] = (stock_data['Next_Day_Return'] > 0).astype(int)
        
        return stock_data[['Close', 'Volume', 'Returns', 'Log_Returns', 'Volatility', 
                          'MA_5', 'MA_20', 'RSI', 'Next_Day_Return', 'Direction']]
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def collect_economic_data(self):
        """Collect macroeconomic data from FRED"""
        print("Collecting macroeconomic data from FRED...")
        
        # Initialize FRED API
        fred = Fred(api_key=self.fred_api_key)
        
        # Economic indicators to collect
        indicators = {
            'GDP': 'GDP',                           # Gross Domestic Product
            'UNEMPLOYMENT': 'UNRATE',               # Unemployment Rate
            'INFLATION': 'CPIAUCSL',                # Consumer Price Index
            'FED_RATE': 'FEDFUNDS',                 # Federal Funds Rate
            'VIX': 'VIXCLS',                        # VIX Volatility Index
            '10Y_TREASURY': 'GS10',                 # 10-Year Treasury Rate
            '3M_TREASURY': 'GS3M',                  # 3-Month Treasury Rate
            'CONSUMER_SENTIMENT': 'UMCSENT',        # Consumer Sentiment
            'INDUSTRIAL_PRODUCTION': 'INDPRO',      # Industrial Production Index
            'HOUSING_STARTS': 'HOUST',              # Housing Starts
            'RETAIL_SALES': 'RSAFS',                # Retail Sales
            'M2_MONEY_SUPPLY': 'M2SL'               # M2 Money Supply
        }
        
        economic_data = pd.DataFrame()
        
        for name, code in indicators.items():
            try:
                print(f"  Fetching {name}...")
                series = fred.get_series(code, start=self.start_date, end=self.end_date)
                
                # Ensure timezone-naive index
                if hasattr(series.index, 'tz') and series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                
                series = pd.DataFrame({name: series})
                
                if economic_data.empty:
                    economic_data = series
                else:
                    economic_data = economic_data.join(series, how='outer')
            except Exception as e:
                print(f"  Warning: Could not fetch {name}: {e}")
        
        # Calculate derived features
        if 'INFLATION' in economic_data.columns:
            economic_data['INFLATION_RATE'] = economic_data['INFLATION'].pct_change(12) * 100
        
        if '10Y_TREASURY' in economic_data.columns and '3M_TREASURY' in economic_data.columns:
            economic_data['YIELD_CURVE'] = economic_data['10Y_TREASURY'] - economic_data['3M_TREASURY']
        
        return economic_data
    
    def prepare_data(self):
        """Combine and prepare all data for modeling"""
        print("Preparing and cleaning data...")
        
        # Collect data
        stock_data = self.collect_stock_data()
        economic_data = self.collect_economic_data()
        
        # Resample economic data to daily frequency (forward fill)
        economic_data_daily = economic_data.resample('D').ffill()
        
        # Combine datasets
        combined_data = stock_data.join(economic_data_daily, how='inner')
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        feature_columns = [col for col in combined_data.columns if col not in ['Next_Day_Return', 'Direction']]
        combined_data[feature_columns] = imputer.fit_transform(combined_data[feature_columns])
        
        # Remove rows with missing target variables
        combined_data = combined_data.dropna(subset=['Next_Day_Return', 'Direction'])
        
        # Add lag features for economic indicators
        econ_columns = [col for col in combined_data.columns if col not in stock_data.columns]
        for col in econ_columns:
            if col in combined_data.columns:
                combined_data[f'{col}_lag1'] = combined_data[col].shift(1)
                combined_data[f'{col}_lag7'] = combined_data[col].shift(7)
        
        # Remove rows with NaN after creating lag features
        combined_data = combined_data.dropna()
        
        self.data = combined_data
        print(f"Final dataset shape: {combined_data.shape}")
        return combined_data
    
    def visualize_data(self):
        """Create visualizations of the data"""
        print("Creating visualizations...")
        
        # Set up the plotting style with better formatting
        plt.style.use('default')
        fig = plt.figure(figsize=(24, 18))
        plt.rcParams.update({'font.size': 10})
        
        # 1. Stock price and moving averages
        plt.subplot(3, 3, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price', alpha=0.7, linewidth=1.5)
        plt.plot(self.data.index, self.data['MA_5'], label='5-day MA', alpha=0.8, linewidth=1)
        plt.plot(self.data.index, self.data['MA_20'], label='20-day MA', alpha=0.8, linewidth=1)
        plt.title(f'{self.stock_symbol} Price and Moving Averages', fontsize=12, fontweight='bold')
        plt.legend(loc='upper left', fontsize=9)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 2. Returns distribution
        plt.subplot(3, 3, 2)
        plt.hist(self.data['Returns'].dropna(), bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        plt.title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Daily Returns', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 3. Volatility over time
        plt.subplot(3, 3, 3)
        plt.plot(self.data.index, self.data['Volatility'], color='orange', alpha=0.7)
        plt.title('Volatility Over Time', fontsize=12, fontweight='bold')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 4. Economic indicators
        plt.subplot(3, 3, 4)
        if 'FED_RATE' in self.data.columns:
            plt.plot(self.data.index, self.data['FED_RATE'], label='Fed Rate', linewidth=1.5)
        if 'UNEMPLOYMENT' in self.data.columns:
            plt.plot(self.data.index, self.data['UNEMPLOYMENT'], label='Unemployment', linewidth=1.5)
        plt.title('Key Economic Indicators', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 5. Yield curve
        plt.subplot(3, 3, 5)
        if 'YIELD_CURVE' in self.data.columns:
            plt.plot(self.data.index, self.data['YIELD_CURVE'], color='purple', alpha=0.7)
            plt.title('Yield Curve (10Y - 3M)', fontsize=12, fontweight='bold')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
            plt.xticks(rotation=45, fontsize=8)
            plt.yticks(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        # 6. VIX
        plt.subplot(3, 3, 6)
        if 'VIX' in self.data.columns:
            plt.plot(self.data.index, self.data['VIX'], color='red', alpha=0.7)
            plt.title('VIX (Fear Index)', fontsize=12, fontweight='bold')
            plt.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='High Fear')
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
            plt.xticks(rotation=45, fontsize=8)
            plt.yticks(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        # 7. Correlation heatmap
        plt.subplot(3, 3, 7)
        correlation_data = self.data.select_dtypes(include=[np.number]).corr()
        # Select most important correlations
        important_features = ['Returns', 'FED_RATE', 'UNEMPLOYMENT', 'VIX', 'YIELD_CURVE', 'INFLATION_RATE']
        available_features = [f for f in important_features if f in correlation_data.columns]
        if len(available_features) > 1:
            corr_subset = correlation_data.loc[available_features, available_features]
            # Create shorter labels for better display
            short_labels = []
            for feature in available_features:
                if feature == 'UNEMPLOYMENT': short_labels.append('UNEMP')
                elif feature == 'INFLATION_RATE': short_labels.append('INFL')
                elif feature == 'YIELD_CURVE': short_labels.append('Y_CURVE')
                elif feature == 'FED_RATE': short_labels.append('FED')
                else: short_labels.append(feature)
            
            corr_subset.index = short_labels
            corr_subset.columns = short_labels
            sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                       cbar_kws={'shrink': 0.8}, square=True)
            plt.title('Feature Correlations', fontsize=12, fontweight='bold')
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
        
        # 8. Direction distribution
        plt.subplot(3, 3, 8)
        direction_counts = self.data['Direction'].value_counts()
        bars = plt.bar(['Down (0)', 'Up (1)'], direction_counts.values, 
                      color=['lightcoral', 'lightgreen'], alpha=0.8, edgecolor='black')
        plt.title('Direction Distribution', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # 9. Returns vs VIX
        plt.subplot(3, 3, 9)
        if 'VIX' in self.data.columns:
            plt.scatter(self.data['VIX'], self.data['Returns'], alpha=0.3, s=8)
            plt.xlabel('VIX', fontsize=10)
            plt.ylabel('Returns', fontsize=10)
            plt.title('Returns vs VIX', fontsize=12, fontweight='bold')
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        # Improve overall layout
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.show()
    
    def prepare_features(self):
        """Prepare features for modeling"""
        # Exclude target variables and non-predictive features
        exclude_columns = ['Next_Day_Return', 'Direction', 'Close']
        feature_columns = [col for col in self.data.columns if col not in exclude_columns]
        
        X = self.data[feature_columns]
        y_reg = self.data['Next_Day_Return']  # For regression
        y_clf = self.data['Direction']        # For classification
        
        return X, y_reg, y_clf, feature_columns
    
    def train_classification_model(self, X, y):
        """Train classification model to predict market direction"""
        print("Training classification models...")
        
        # Split data maintaining temporal order
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['classification'] = scaler
        
        # Train Random Forest
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_clf.fit(X_train_scaled, y_train)
        
        # Train Logistic Regression
        lr_clf = LogisticRegression(random_state=42, max_iter=1000)
        lr_clf.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_pred = rf_clf.predict(X_test_scaled)
        lr_pred = lr_clf.predict(X_test_scaled)
        
        rf_score = rf_clf.score(X_test_scaled, y_test)
        lr_score = lr_clf.score(X_test_scaled, y_test)
        
        print(f"Random Forest Accuracy: {rf_score:.4f}")
        print(f"Logistic Regression Accuracy: {lr_score:.4f}")
        
        # Choose best model
        if rf_score > lr_score:
            best_model = rf_clf
            best_pred = rf_pred
            model_name = "Random Forest"
        else:
            best_model = lr_clf
            best_pred = lr_pred
            model_name = "Logistic Regression"
        
        self.models['classification'] = {
            'model': best_model,
            'name': model_name,
            'score': max(rf_score, lr_score)
        }
        
        print(f"\nBest Classification Model: {model_name}")
        print("\nClassification Report:")
        print(classification_report(y_test, best_pred))
        
        return X_test, y_test, best_pred
    
    def train_regression_model(self, X, y):
        """Train regression model to predict returns"""
        print("\nTraining regression models...")
        
        # Split data maintaining temporal order
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['regression'] = scaler
        
        # Train Random Forest Regressor
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_reg.fit(X_train_scaled, y_train)
        
        # Train Linear Regression
        lr_reg = LinearRegression()
        lr_reg.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_pred = rf_reg.predict(X_test_scaled)
        lr_pred = lr_reg.predict(X_test_scaled)
        
        rf_r2 = r2_score(y_test, rf_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        
        rf_mse = mean_squared_error(y_test, rf_pred)
        lr_mse = mean_squared_error(y_test, lr_pred)
        
        print(f"Random Forest - R²: {rf_r2:.4f}, MSE: {rf_mse:.6f}")
        print(f"Linear Regression - R²: {lr_r2:.4f}, MSE: {lr_mse:.6f}")
        
        # Choose best model based on R²
        if rf_r2 > lr_r2:
            best_model = rf_reg
            best_pred = rf_pred
            model_name = "Random Forest"
            best_r2 = rf_r2
        else:
            best_model = lr_reg
            best_pred = lr_pred
            model_name = "Linear Regression"
            best_r2 = lr_r2
        
        self.models['regression'] = {
            'model': best_model,
            'name': model_name,
            'r2': best_r2
        }
        
        print(f"\nBest Regression Model: {model_name}")
        
        return X_test, y_test, best_pred
    
    def feature_importance(self, feature_columns):
        """Display feature importance"""
        print("\nFeature Importance Analysis:")
        
        # Classification feature importance
        if hasattr(self.models['classification']['model'], 'feature_importances_'):
            clf_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.models['classification']['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 features for {self.models['classification']['name']} (Classification):")
            print(clf_importance.head(10))
        
        # Regression feature importance
        if hasattr(self.models['regression']['model'], 'feature_importances_'):
            reg_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.models['regression']['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 features for {self.models['regression']['name']} (Regression):")
            print(reg_importance.head(10))
    
    def make_predictions(self, days_ahead=5):
        """Make future predictions"""
        print(f"\nMaking predictions for the next {days_ahead} days...")
        
        # Get the latest data
        latest_features = self.data.iloc[-1][self.feature_columns].values.reshape(1, -1)
        
        # Scale the features
        clf_features_scaled = self.scalers['classification'].transform(latest_features)
        reg_features_scaled = self.scalers['regression'].transform(latest_features)
        
        # Make predictions
        direction_pred = self.models['classification']['model'].predict(clf_features_scaled)[0]
        return_pred = self.models['regression']['model'].predict(reg_features_scaled)[0]
        
        direction_prob = None
        if hasattr(self.models['classification']['model'], 'predict_proba'):
            direction_prob = self.models['classification']['model'].predict_proba(clf_features_scaled)[0]
        
        print(f"Next day direction prediction: {'UP' if direction_pred == 1 else 'DOWN'}")
        if direction_prob is not None:
            print(f"Probability - Down: {direction_prob[0]:.3f}, Up: {direction_prob[1]:.3f}")
        print(f"Next day return prediction: {return_pred:.4f} ({return_pred*100:.2f}%)")
        
        return direction_pred, return_pred, direction_prob
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Stock Market Prediction Analysis...")
        print("=" * 50)
        
        # Prepare data
        self.prepare_data()
        
        # Visualize data
        self.visualize_data()
        
        # Prepare features
        X, y_reg, y_clf, feature_columns = self.prepare_features()
        self.feature_columns = feature_columns
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {len(feature_columns)}")
        
        # Train models
        X_test_clf, y_test_clf, y_pred_clf = self.train_classification_model(X, y_clf)
        X_test_reg, y_test_reg, y_pred_reg = self.train_regression_model(X, y_reg)
        
        # Feature importance
        self.feature_importance(feature_columns)
        
        # Make predictions
        self.make_predictions()
        
        print("\nAnalysis completed successfully!")
        
        return self.models, self.data

# Usage Example
if __name__ == "__main__":
    # Initialize the predictor
    fred_api_key = "4e0ac07a1fa873e5599d209f5a3465b9"  
    
    predictor = StockMarketPredictor(
        fred_api_key=fred_api_key,
        stock_symbol='SPY',  # S&P 500 ETF
        start_date='2015-01-01'
    )
    
    # Run the full analysis
    try:
        models, data = predictor.run_full_analysis()
        
        # You can access the trained models and data
        print(f"\nClassification Model: {models['classification']['name']}")
        print(f"Classification Accuracy: {models['classification']['score']:.4f}")
        print(f"Regression Model: {models['regression']['name']}")
        print(f"Regression R²: {models['regression']['r2']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")

