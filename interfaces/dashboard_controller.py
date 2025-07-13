import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output


class DashboardController:
    def __init__(self, neuro_engine):
        self.app = dash.Dash(__name__)
        self.engine = neuro_engine
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Настройка макета дашборда"""
        self.app.layout = html.Div([
            html.H1("NeuroCrypto Nexus Dashboard"),
            dcc.Dropdown(
                id='symbol-selector',
                options=[
                    {'label': 'BTC/USDT', 'value': 'BTC/USDT'},
                    {'label': 'ETH/USDT', 'value': 'ETH/USDT'},
                    {'label': 'BNB/USDT', 'value': 'BNB/USDT'}
                ],
                value='BTC/USDT'
            ),
            dcc.Graph(id='price-chart'),
            dcc.Graph(id='sentiment-chart'),
            dcc.Interval(id='update-interval', interval=60 * 1000)
        ])

    def setup_callbacks(self):
        """Настройка интерактивных элементов"""

        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('sentiment-chart', 'figure')],
            [Input('update-interval', 'n_intervals'),
             Input('symbol-selector', 'value')]
        )
        def update_charts(n, symbol):
            # Получение данных от нейросистемы
            price_data = self.engine.get_market_data(symbol)
            sentiment_data = self.engine.get_sentiment_data(symbol)

            # График цен
            price_fig = {
                'data': [go.Scatter(
                    x=price_data['timestamps'],
                    y=price_data['prices'],
                    name='Price'
                )],
                'layout': go.Layout(title=f'{symbol} Price')
            }

            # График настроений
            sentiment_fig = {
                'data': [go.Bar(
                    x=sentiment_data['sources'],
                    y=sentiment_data['scores'],
                    name='Sentiment'
                )],
                'layout': go.Layout(title=f'{symbol} Sentiment Analysis')
            }

            return price_fig, sentiment_fig

    def run_server(self):
        """Запуск сервера дашборда"""
        self.app.run_server(host='0.0.0.0', port=8050, debug=False)