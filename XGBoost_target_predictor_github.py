import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import warnings
import time
from collections import Counter
warnings.filterwarnings('ignore')

# Default parameters have been initialised here, you can tweak it based on your strategy
SMA_WINDOW = 20           
VOLUME_WINDOW = 10        
VOLATILITY_WINDOW = 14    
RSI_WINDOW = 14           
VWAP_WINDOW = 10          
TARGET_WINDOW = 10        
ATR_WINDOW = 14           
ATR_MULTIPLIER = 2          
SCALING_WINDOW = 20       

# Getting user inputs (preffered stocks)
raw_tickers = input("Enter the Ticker(s) separated by commas (e.g. RELIANCE.NS, TCS.NS, INFY.NS): ")
tickers = [t.strip().upper() for t in raw_tickers.split(',')]

print(f"\n⚠️ SYSTEM NOTE: You have queued {len(tickers)} stock(s). More stocks will proportionally increase the training time.")

#Allowing user to choose the threshold of each stock if its an invalid entry then we automatically shift 
# to default threshold of 0.50 which is just probability of getting heads or tails when we flip a fair coin
user_threshold = input("Enter your chosen Global Threshold for all stocks (e.g., 0.50, 0.55): ")
try:
    threshold = float(user_threshold)
except ValueError:
    print("Invalid threshold. Defaulting to 0.50")
    threshold = 0.50

print("\nFetching global market data (^NSEI) from Yahoo Finance...")
nifty = yf.download("^NSEI", period='10y', progress=False)

#Often Yahoo Finance gives us a Multi Index data and our mode might confused with two levels showing same data so we drop 
# the first level
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.droplevel(1)
nifty_return = nifty['Close'].pct_change()

master_results = []
all_winning_settings = []

print("\n" + "="*70)
print(f" 🚀 INITIATING ALGORITHMIC SCAN FOR {len(tickers)} ASSET(S)")
print("="*70)

#Running our algorithm for a bunch of stocks entered by the user
for ticker in tickers:
    try:
        print(f"\n[{time.strftime('%H:%M:%S')}] Booting Engine for {ticker}...")
        
        df = yf.download(ticker, period='10y', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        #We cant compare raw price move of a stock with price move of our base index Nifty so instead we use there Percent Change.
        df_return = df['Close'].pct_change()
        
        # INDICATOR-1 : Relative Strength of our stock with respect to Nifty in order to know did it outperform or underperform Nifty
        raw_rel_strength = (df_return - nifty_return)
        df['Relative_Strength'] = (raw_rel_strength - raw_rel_strength.rolling(SCALING_WINDOW).mean()) / raw_rel_strength.rolling(SCALING_WINDOW).std()

        # INDICATOR-2 : Current distance (spread) of our stock from its 50-DAY Simple Moving Average
        df['Dist_from_50_SMA'] = (df['Close'] / df['Close'].rolling(window=SMA_WINDOW).mean()) - 1
        
        # INDICATOR-3 : Surge in the Volume of stock as compared to the defined window of past days
        raw_vol_surge = (df['Volume'] / df['Volume'].rolling(window=VOLUME_WINDOW).mean()) - 1
        df[f'Volume_Surge_{VOLUME_WINDOW}'] = (raw_vol_surge - raw_vol_surge.rolling(SCALING_WINDOW).mean()) / raw_vol_surge.rolling(SCALING_WINDOW).std()

        # INDICATOR-4 : How much does the stock move daily with respect to its daily avg closing price, 
        # basically how much does the stock deviate from its daily expected position
        df[f'Volatility_{VOLATILITY_WINDOW}'] = df['Close'].pct_change().rolling(window=VOLATILITY_WINDOW).std()
        
        # INDICATOR-5 : Bollinger Band Width basically measures the squeeze of a stock just like a spring, if a stock is moving 
        # in extremely squeezed zone then it migh be getting close to just jump and release that momentum in a move just like
        #  a compressed spring
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        df['BB_Width'] = (upper_band - lower_band) / rolling_mean
        
        # INDICATOR-6 : MACD Historgram measures the accelerating momentum of a stock, like if histogram is below 0 and starts 
        # printing smaller and smaller red bars then it signifies that the selling momentum is dying and if it starts making 
        # small green bars above 0 it means buying momentum is about to begin
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = macd - signal
        
        # INDICATOR-7 : Rate of change calculates the exact percent change in a stock in defined period of time, 
        # if the stock is moving up slowly , ROC stays low, if explodes towards upside then ROC increases sharply 
        # indicating a stong short momentum behind the stock and same when the stock is crashing, ROC bleeds sharply
        df['ROC_3'] = df['Close'].pct_change(periods=3)
        
        # INDICATOR-8 : VWAP is the true fair price of the stock where big instituional whales love to buy a stock at.
        # If VWAP is highly positive, it signifies the stock is overextended and people are buying it , if the deviation is 0
        # the stock is sitting at the VWAP Line which is the fair price institutions want to get in at.
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        rolling_vp = (typical_price * df['Volume']).rolling(window=VWAP_WINDOW).mean()
        rolling_v = df['Volume'].rolling(window=VWAP_WINDOW).mean()
        df['Rolling_VWAP'] = rolling_vp / rolling_v
        
        df['VWAP_Deviation'] = (df['Close'] - df['Rolling_VWAP']) / df['Rolling_VWAP']

        # INDICATOR-9 : RSI just measures the no. of daily wins / no. of daily loses in the defined period
        # if wins are more then RSI would be higher signalling the stock is overbought and if loses are more the RSI would be lower
        #signalling that the stock is oversold.
        delta = df["Close"].diff()
        gains = delta.where(delta > 0, 0).rolling(window=RSI_WINDOW).mean()
        loses = abs(delta.where(delta < 0, 0).rolling(window=RSI_WINDOW).mean())
        rel_str = gains / loses
        df[f'RSI_{RSI_WINDOW}'] = (100 - (100 / (1 + rel_str)))

        # DYNAMIC TARGET CREATION (ATR) 
        # Average True Range : It basically calculates target based on a the stocks daily volatility (how much stock moves on an avg)
        # 1.Calculate True Range (TR)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # 2.Calculate ATR (14-day rolling average of True Range)
        df['ATR'] = true_range.rolling(window=ATR_WINDOW).mean()
        
        # 3.Calculate Dynamic Target
        # Instead of a fixed 2%, the target price is the Close + (ATR * Multiplier)
        future_highs = df['High'].rolling(window=TARGET_WINDOW).max().shift(-TARGET_WINDOW)
        target_price = df['Close'] + (df['ATR'] * ATR_MULTIPLIER)
        
        # 4.Generate Target Signal (1 if target hit, 0 if not)
        df['Target'] = (future_highs > target_price).astype(int)
        

        df = df.dropna()
        features = [
            'Relative_Strength', 'Dist_from_50_SMA', f'Volume_Surge_{VOLUME_WINDOW}', 
            f'Volatility_{VOLATILITY_WINDOW}', f'RSI_{RSI_WINDOW}', 'BB_Width', 'MACD_Hist', 'ROC_3', 'VWAP_Deviation'
        ]

        X = df[features]
        Y = df['Target']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

        # TRAINING THE MODEL
        # 1.Finding the natural class imbalance
        zero_count = len(Y_train[Y_train == 0])
        one_count = len(Y_train[Y_train == 1])
        base_imbalance = zero_count / one_count if one_count > 0 else 1
        
        # 2.Adding an Aggression Multiplier to penalize the model for missing breakouts
        # (1.5 to 2.0 forces the model to take more risks and hunt for 1s (Wins))
        AGGRESSION_FACTOR = 1.0
        hunting_weight = base_imbalance * AGGRESSION_FACTOR
        
        base_xgb = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=1, scale_pos_weight=hunting_weight)
        
        # Generic Search Space for Public Release
        search_space = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'reg_alpha': [0, 1],
            'reg_lambda': [1, 2]
        }
        
        #Using Randomized Search CV to make multiple combinatiions (a total of 20 as n_iter = 20) from our given search space 
        # and find the best combination of parameters
        XGB_Search = RandomizedSearchCV(estimator=base_xgb, param_distributions=search_space, n_iter=5, cv=TimeSeriesSplit(n_splits=5), random_state=42)
        XGB_Search.fit(X_train, Y_train)
        best_XGB = XGB_Search.best_estimator_

        #EVALUATION & RANKING SYSTEM
        probabilities = best_XGB.predict_proba(X_test)
        predictions = (probabilities[:, 1] >= threshold).astype(int)
        
        acc = accuracy_score(Y_test, predictions)
        report = classification_report(Y_test, predictions, output_dict=True, zero_division=0)
        
        win_rate = report['1']['precision'] if '1' in report else 0
        caught = report['1']['recall'] if '1' in report else 0
        total_ops = int(report['1']['support']) if '1' in report else 0

        #LIVE SIGNALS TO TRADE
        today_proba = best_XGB.predict_proba(X.tail(1))[0][1]
        
        if today_proba >= (threshold + 0.15):
            live_signal = "🔥 STRONG BUY"
            rank = 1

        elif today_proba >= threshold:
            live_signal = "🟢 BUY"
            rank = 2

        elif today_proba >= (threshold - 0.10):
            live_signal = "🟡 WEAK BUY"
            rank = 3

        else:
            live_signal = "🔴 WAIT"
            rank = 4

        #Calculating the actual percentage target for the live signal
        current_close = df['Close'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        target_pct_move = ((current_atr * ATR_MULTIPLIER) / current_close) * 100
        best_params_str = f"Dep:{XGB_Search.best_params_['max_depth']} | Est:{XGB_Search.best_params_['n_estimators']} | LR:{XGB_Search.best_params_['learning_rate']}"
        all_winning_settings.append(best_params_str)
        
        #Storing all the results
        master_results.append({
            "Ticker": ticker,
            "Target %": f"{target_pct_move:.2f}%", 
            "Accuracy": f"{acc * 100:.1f}%",
            "Win Rate": f"{win_rate * 100:.1f}%",
            "Win_Raw": win_rate,
            "Caught": f"{caught * 100:.1f}%",
            "Opps": total_ops,
            "Live Signal": live_signal,
            "Conf %": f"{today_proba * 100:.1f}%",
            "Conf_Raw": today_proba,
            "Rank": rank,
            "Engine Settings": best_params_str
        })

        # We use matplotlib to plot te feature importance %s only when 1 stock is entered 
        if len(tickers) == 1:
            feature_importance_scores = best_XGB.feature_importances_
            sorted_scores = pd.Series(feature_importance_scores, index=features).sort_values(ascending=False)
            plt.figure(figsize=(12,6))
            sorted_scores.plot(kind='bar', color='green')
            plt.title(f"Feature Importance scores for : {ticker} (XGBoost)") 
            plt.xlabel("Predictive Indicators")
            plt.ylabel("Importance Score %")
            plt.xticks(rotation=45)
            plt.grid(linestyle='--', axis='y')
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

#OUTPUT TABLE
if master_results:
    print("\n\n" + "="*100)
    print(" 🏆 TERMINAL OUTPUT: MASTER BATCH SUMMARY ")
    print("="*100)
    
    results_df = pd.DataFrame(master_results)
    
    #Sorting by Rank (1 to 4), then by Win Rate (Descending), then by Confidence
    results_df = results_df.sort_values(by=['Rank', 'Win_Raw', 'Conf_Raw'], ascending=[True, False, False])
    
    #Dropping the hidden raw columns before displaying
    results_df = results_df.drop(columns=['Rank', 'Win_Raw', 'Conf_Raw'])
    results_df.set_index('Ticker', inplace=True)
    
    #Using markdown to create a professional grid format
    print(results_df.to_markdown(tablefmt="grid"))
    print("="*100 + "\n")

    #Getting the best settings (alpha) used by our model 
    if all_winning_settings:
        most_common = Counter(all_winning_settings).most_common(1)[0]
        print(" 👑 THE OPTIMIZED BATCH ALPHA SETTING")
        print("="*100)
        print(f"The most dominant engine setup across this basket of stocks was:")
        print(f"➡️  [ {most_common[0]} ]")
        print(f"This specific configuration won the randomized search on {most_common[1]} out of {len(tickers)} stocks.")
        print("="*100 + "\n")
else:
    print("\n❌ No stocks were successfully processed.")