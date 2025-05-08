#!/usr/bin/env python3

import warnings
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # suppress ARIMA warnings

def main():
    # 1) Load & prepare monthly data
    df = pd.read_csv(
        'Merged_AQI_Income_Poverty_Unemployment_Livability.csv',
        parse_dates=['Date']
    )
    df['year_month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    monthly = (
        df.groupby(['County Name', 'year_month'])
          .agg({
              'AQI': 'mean',
              'Median_Income': 'mean',
              'Poverty_Rate': 'mean',
              'Unemployment_Rate': 'mean'
          })
          .reset_index()
          .sort_values(['County Name', 'year_month'])
    )
    
    test_size = 6
    results = []
    
    for county, group in monthly.groupby('County Name'):
        if len(group) <= test_size:
            continue
        
        grp = group.reset_index(drop=True)
        grp['AQI_lag1'] = grp['AQI'].shift(1)
        grp = grp.dropna().reset_index(drop=True)
        if len(grp) <= test_size:
            continue
        
        train = grp.iloc[:-test_size]
        test  = grp.iloc[-test_size:]
        
        feats = ['AQI_lag1','Median_Income','Poverty_Rate','Unemployment_Rate']
        X_train, y_train = train[feats], train['AQI']
        X_test,  y_test  = test[feats],  test['AQI']
        
        # Baseline
        baseline = DummyRegressor(strategy='mean').fit(X_train, y_train)
        y_base = baseline.predict(X_test)
        rmse_base = np.sqrt(mean_squared_error(y_test, y_base))
        
        # ARIMAX
        try:
            arimax = ARIMA(
                endog=y_train,
                exog=X_train[['Median_Income','Poverty_Rate','Unemployment_Rate']],
                order=(1,1,1)
            ).fit()
            y_arimax = arimax.predict(
                start=test.index[0], end=test.index[-1],
                exog=X_test[['Median_Income','Poverty_Rate','Unemployment_Rate']]
            )
            rmse_arimax = np.sqrt(mean_squared_error(y_test, y_arimax))
        except Exception:
            rmse_arimax = np.nan
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        y_rf = rf.predict(X_test)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_rf))
        
        results.append({
            'County': county,
            'Baseline_RMSE': rmse_base,
            'ARIMAX_RMSE': rmse_arimax,
            'RF_RMSE': rmse_rf
        })
        
        print(f"{county} | Baseline RMSE: {rmse_base:.2f} | "
              f"ARIMAX RMSE: {rmse_arimax:.2f} | RF RMSE: {rmse_rf:.2f}")
    
    # summary
    res_df = pd.DataFrame(results)
    print("\n===== Average RMSE across counties =====")
    for c in ['Baseline_RMSE','ARIMAX_RMSE','RF_RMSE']:
        print(f"{c}: {res_df[c].mean():.2f}")

if __name__ == "__main__":
    main()
