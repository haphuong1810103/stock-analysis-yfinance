from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, date, timedelta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def parse_date(date_string):
    try:
        return datetime.strptime(date_string, '%Y-%m-%d').date()
    except ValueError:
        try:
            return datetime.strptime(date_string, '%m/%d/%Y').date()
        except ValueError:
            raise ValueError(f"Unable to parse date: {date_string}. Please use YYYY-MM-DD or MM/DD/YYYY format.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    symbol = request.form['symbol']
    option = request.form['option']
    
    try:
        if option == 'A':
            return get_price_chart(symbol)
        elif option == 'B':
            symbol2 = request.form['symbol2']
            return get_price_comparison(symbol, symbol2)
        elif option == 'C':
            return get_revenue_earnings(symbol)
        elif option == 'D':
            return get_cash_flow(symbol)
        elif option == 'E':
            return get_analyst_recommendations(symbol)
        elif option == 'F':
            return get_price_prediction(symbol)
        else:
            return jsonify({'error': 'Invalid option'})
    except Exception as e:
        return jsonify({'error': str(e)})

def get_price_chart(symbol):
    try:
        sdate = parse_date(request.form['start_date'])
        edate = parse_date(request.form['end_date'])
    except ValueError as e:
        return jsonify({'error': str(e)})
    
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=sdate, end=edate)
        
        if df.empty:
            return jsonify({'error': f"No data available for {symbol} in the specified date range."})
        
        df = df.reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.Date, y=df['Close'], line_color='deepskyblue', name='Close Price'))
        fig.update_layout(title=f'{symbol} Stock Price from {sdate} to {edate}',
                          yaxis_title='Closing Price in USD',
                          xaxis_tickfont_size=14,
                          yaxis_tickfont_size=14)
        
        return jsonify({'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'error': f"Error retrieving data for {symbol}: {str(e)}"})

def get_price_comparison(symbol, symbol2):
    try:
        sdate = parse_date(request.form['start_date'])
        edate = parse_date(request.form['end_date'])
    except ValueError as e:
        return jsonify({'error': str(e)})
    
    try:
        stock1 = yf.Ticker(symbol)
        stock2 = yf.Ticker(symbol2)
        
        df1 = stock1.history(start=sdate, end=edate)
        df2 = stock2.history(start=sdate, end=edate)
        
        if df1.empty or df2.empty:
            return jsonify({'error': f"No data available for one or both stocks in the specified date range."})
        
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df1.Date, y=df1['Close'], name=f'{symbol} Stock', line_color='deepskyblue', opacity=0.9))
        fig.add_trace(go.Scatter(x=df2.Date, y=df2['Close'], name=f'{symbol2} Stock', line_color='dimgray', opacity=0.9))
        
        fig.update_layout(title=f'Price Comparison of {symbol} Stock and {symbol2} Stock from {sdate} to {edate}',
                          yaxis_title='Closing Price in USD',
                          xaxis_tickfont_size=14,
                          yaxis_tickfont_size=14)
        
        return jsonify({'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'error': f"Error retrieving data: {str(e)}"})

def get_revenue_earnings(symbol):

    # get the net income and total revenue stacked bar chart 
    try:
        stock = yf.Ticker(symbol)
        income_statement = stock.income_stmt
        net_income_revenue = income_statement.loc[["Net Income", "Total Revenue"]]
        if income_statement.empty:
            return jsonify({'error': f"No earnings data available for {symbol}."})
        
        net_income_revenue_df = net_income_revenue.T.reset_index()
        net_income_revenue_df.columns = ["year", "total_revenue", "net_income"]

        net_income_revenue_df['year'] = net_income_revenue_df['year'].dt.year
        net_income_revenue_df = net_income_revenue_df.sort_values(by="year", ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=net_income_revenue_df['year'], y=net_income_revenue_df['total_revenue'], name='Total Revenue', marker_color='rgb(55, 83, 109)'))
        fig.add_trace(go.Bar(x=net_income_revenue_df['year'], y=net_income_revenue_df['net_income'], name='Net Income', marker_color='rgb(26, 118, 255)'))
        
        fig.update_layout(title=f'Total Revenue and Net Income of {symbol} Stock',
                          yaxis_title='USD',
                          xaxis_tickfont_size=14,
                          yaxis_tickfont_size=14,
                          barmode='group',
                          bargap=0.15,
                          bargroupgap=0.1)
        
        return jsonify({'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'error': f"Error retrieving data for {symbol}: {str(e)}"})

def get_cash_flow(symbol):
    try:
        stock = yf.Ticker(symbol)
        cashflow = stock.cashflow
        
        if cashflow.empty:
            return jsonify({'error': f"No cash flow data available for {symbol}."})
        
        cashflow = cashflow.rename(columns=lambda x: str(x.year))

        ICF = cashflow.loc['Investing Cash Flow',]
        FCF = cashflow.loc['Financing Cash Flow',]
        OCF = cashflow.loc['Operating Cash Flow',]
        NI = cashflow.loc['Net Income From Continuing Operations',]

        # We save the variables in "CF"
        CF = [OCF, ICF, FCF, NI] 

        # With this data we create a new dataframe "cashflow"
        cashflow = pd.DataFrame(CF)
        cashflow = cashflow.reset_index()
                        
            # Change the columnnames and some values within the dataframe 
        cashflow = cashflow.rename(columns={'index': 'Cashflows'})
        cashflow.replace('Operating Cash Flow', 'Cash Flow From Operating Activities', inplace=True)
        cashflow.replace('Investing Cash Flow', 'Cash Flow From Investing Activities', inplace=True)
        cashflow.replace('Financing Cash Flow', 'Cash Flow From Financing Activities', inplace=True)
        cashflow.replace('Net Income From Continuing Operations', 'Net Income', inplace=True)
        
        # Save the data of the different columns in separate variables, due to the future changes of the years or data
        first_cf = cashflow.iloc[:,1]
        second_cf = cashflow.iloc[:,2]
        third_cf = cashflow.iloc[:,3]
        fourth_cf = cashflow.iloc[:,4]


        # Plotting the bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
                x = cashflow.Cashflows,
                y = first_cf,
                name = first_cf.name))
        
        fig.add_trace(go.Bar(
                x = cashflow.Cashflows,
                y = second_cf,
                name = second_cf.name))
        
        fig.add_trace(go.Bar(
                x = cashflow.Cashflows,
                y = third_cf,
                name = third_cf.name))

        fig.add_trace(go.Bar(
                x = cashflow.Cashflows,
                y = fourth_cf,
                name = fourth_cf.name))    


        fig.update_layout(barmode = 'group',
                            bargap = 0.15,
                            bargroupgap = 0.1,
                            title = f'Cash Flow Statement of {symbol} Stock',
                            xaxis_tickfont_size = 14,
                            xaxis_tickangle = -20,
                            yaxis = dict(
                                    title = 'USD (billions)',
                                    titlefont_size = 14,
                                    tickfont_size = 12))
    
        return jsonify({'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'error': f"Error retrieving data for {symbol}: {str(e)}"})

def get_analyst_recommendations(symbol):
    try:
        

        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations

        valid_periods = recommendations['period'].apply(lambda x: int(x.strip('m')) if 'm' in x else 0)
        recommendations = recommendations[valid_periods >= -6] 
        
        if recommendations.empty:
            return jsonify({'error': f"No analyst recommendations available for {symbol} in the last 6 months."})
        
        
        recommendation_sums = recommendations[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']].sum()

        labels = recommendation_sums.index
        values = recommendation_sums.values
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title=f'Analyst Recommendations for {symbol} Stock (Last 6 Months)')
        
        return jsonify({'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'error': f"Error retrieving data for {symbol}: {str(e)}"})

def get_price_prediction(symbol):
    try:

        today = date.today()
        today = today.strftime('%Y-%m-%d')

        # Get the stock data, starting from 10 years ago
        start_date = datetime.strptime(today, '%Y-%m-%d') - timedelta(days=365*10)
        df = yf.download(symbol, start=start_date, end=today)

        df = df[['Adj Close']]
        
        if df.empty:
            return jsonify({'error': f"No historical data available for {symbol}."})
        
        n = 30

        # Create another column "Prediction" shifted "n" units up
        df['Prediction'] = df[['Adj Close']].shift(-n)
        # We shifted the data up 30 rows, so that for every date we have the actual price ("Adj Close") and the predicted price 30 days into the future ("Prediction") 
        # Therefore the last 30 rows of the column "Prediction" will be empty or contain the value "NaN"

        X = df.drop(['Prediction'],axis=1)
        # Convert the data into a numpy array 
        X = np.array(X)
        # Remove the last "n" rows
        X = X[:-n]

        # Create the dependent data set "Y"
        # For the dependent data we need the column "Prediction"
        Y = df['Prediction']
        # Convert the data into a numpy array 
        Y = np.array(Y)
        # Remove the last "n" rows
        Y = Y[:-n]
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        
        x_forecast = np.array(df.drop(['Prediction'], axis=1))[-n:]
        lr_prediction = lr.predict(x_forecast)
        
        predictions = pd.DataFrame(lr_prediction, columns = ['Prediction'])
        
        df = df.reset_index()

        # From "Date" we need the to get the last value which is the latest date and add 1 day, because that's the date when our predictions start
        d = df['Date'].iloc[-1]
        d = d + timedelta(days=1)

        # Now we make a list with the respective daterange, beginning from the startdate of our predictions and ending 30 days after
        datelist = pd.date_range(d, periods = 30).tolist()
        # We add the variable to our Dataframe "predictions"
        predictions['Date'] = datelist

        # Save the date of today 6 months ago, by subtracting 6 months from the date of today
        six_months = date.today() - timedelta(days=180)
        six_months = six_months.strftime('%Y-%m-%d')

        # Get the data for plotting
        df = yf.download(symbol, start=six_months, end=today)
        df = df.reset_index()


        fig = go.Figure()

        fig.add_trace(go.Scatter(
                        x=df.Date,
                        y=df['Adj Close'],
                        name=f'{symbol} stock',
                        line_color='deepskyblue',
                        opacity=0.9))
        
        # Add the data from the predictions
        fig.add_trace(go.Scatter(
                        x=predictions.Date,
                        y=predictions['Prediction'],
                        name=f'Prediction',
                        line=dict(color='red', dash = 'dot'),
                        opacity=0.9))
    
        
        fig.update_layout(title=f'Stock Forecast of {symbol} Stock for the next 30 days',
                          yaxis_title='Closing Price in USD',
                          xaxis_tickfont_size=14,
                          yaxis_tickfont_size=14)
        
        return jsonify({'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'error': f"Error predicting prices for {symbol}: {str(e)}"})
# Update the other functions (get_price_comparison, get_revenue_earnings, etc.) similarly

if __name__ == '__main__':
    app.run(debug=True)
