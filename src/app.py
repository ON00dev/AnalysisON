import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import requests
import time


class TradeAnalyzer:
    def __init__(self, symbol, api_key, interval='1min', take_profit=None, stop_loss=None, leverage=None, volume=None,
                 desired_risk_reward_ratio=2.0, db_path='trade_data.db', model=None, cache_duration=60):
        self.symbol = symbol
        self.api_key = api_key
        self.interval = interval
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.leverage = leverage
        self.volume = volume
        self.desired_risk_reward_ratio = desired_risk_reward_ratio
        self.db_path = db_path
        self.model = model
        self.cache_duration = cache_duration  # duração do cache em segundos

        # Inicializa o banco de dados
        self.initialize_database()

    def initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                entry_price REAL,
                take_profit REAL,
                stop_loss REAL,
                volume REAL,
                leverage REAL,
                success INTEGER
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS price_cache (
                symbol TEXT,
                interval TEXT,
                timestamp TEXT,
                price REAL,
                fetched_at REAL
            )
        ''')
        conn.commit()
        conn.close()

    def save_trade(self, entry_price, take_profit, stop_loss, volume, leverage, success):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO trades (entry_price, take_profit, stop_loss, volume, leverage, success)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (entry_price, take_profit, stop_loss, volume, leverage, success))
        conn.commit()
        conn.close()

    def calculate_risk(self, entry_price, stop_loss, volume, leverage):
        return (entry_price - stop_loss) * volume * leverage

    def calculate_reward(self, entry_price, take_profit, volume, leverage):
        return (take_profit - entry_price) * volume * leverage

    def risk_reward_ratio(self, entry_price, take_profit, stop_loss, volume, leverage):
        risk = self.calculate_risk(entry_price, stop_loss, volume, leverage)
        reward = self.calculate_reward(entry_price, take_profit, volume, leverage)
        if risk == 0:
            return None
        return reward / risk

    def suggested_trade(self, ratio):
        if ratio is None or ratio <= 1:
            return "Negativo: A relação risco/recompensa é desfavorável."
        else:
            return "Positivo: A relação risco/recompensa é favorável."

    def optimize_take_profit_stop_loss(self, entry_price, stop_loss, volume, leverage, desired_risk_reward_ratio):
        risk = self.calculate_risk(entry_price, stop_loss, volume, leverage)
        new_take_profit = entry_price + (risk * desired_risk_reward_ratio) / (volume * leverage)
        new_stop_loss = entry_price - risk / (volume * leverage)
        return new_take_profit, new_stop_loss

    def predict_trade_success(self, entry_price, take_profit, stop_loss, volume, leverage):
        if self.model is None:
            return "Modelo não fornecido para previsão."

        features = np.array([[entry_price, take_profit, stop_loss, volume, leverage]])
        features_scaled = self.model['scaler'].transform(features)
        probability = self.model['classifier'].predict_proba(features_scaled)[0][1]
        return probability

    def update_model(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT entry_price, take_profit, stop_loss, volume, leverage, success FROM trades')
        data = c.fetchall()
        conn.close()

        if len(data) == 0:
            return

        data = np.array(data)
        X = data[:, :-1]
        y = data[:, -1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        self.model = {
            'classifier': model,
            'scaler': scaler
        }

    def forecast_price(self, periods=30):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT rowid, entry_price FROM trades ORDER BY rowid')
        data = c.fetchall()
        conn.close()

        if len(data) == 0:
            return None

        df = pd.DataFrame(data, columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'], unit='D')

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def clean_cache(self):
        # Remove dados do cache que expiraram
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            DELETE FROM price_cache WHERE fetched_at < ?
        ''', (time.time() - self.cache_duration,))
        conn.commit()
        conn.close()

    def fetch_latest_price(self):
        # Primeiro, limpar o cache expirado
        self.clean_cache()

        # Verificar cache antes de fazer a chamada à API
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT price, fetched_at FROM price_cache 
            WHERE symbol = ? AND interval = ? 
            ORDER BY fetched_at DESC LIMIT 1
        ''', (self.symbol, self.interval))
        row = c.fetchone()
        conn.close()

        if row:
            cached_price, fetched_at = row
            if time.time() - fetched_at < self.cache_duration:
                return cached_price

        # Fazer chamada à API se cache estiver expirado ou não existir
        url = f'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol={self.symbol}&market=USD&interval={self.interval}&apikey={self.api_key}'
        response = requests.get(url)
        data = response.json()

        if 'Time Series Crypto (' + self.interval + ')' not in data:
            raise ValueError("Erro ao obter os dados da API. Verifique o símbolo e a chave API.")

        latest_timestamp = list(data['Time Series Crypto (' + self.interval + ')'].keys())[0]
        latest_price = float(data['Time Series Crypto (' + self.interval + ')'][latest_timestamp]['4. close'])

        # Armazenar no cache
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO price_cache (symbol, interval, timestamp, price, fetched_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (self.symbol, self.interval, latest_timestamp, latest_price, time.time()))
        conn.commit()
        conn.close()

        return latest_price

    def analyze_trade(self):
        # Atualiza o preço de entrada com o preço mais recente
        entry_price = self.fetch_latest_price()

        # Caso o usuário não tenha definido volume ou outros parâmetros, definir valores padrão ou calcular
        if self.volume is None:
            self.volume = 100  # Exemplo de valor padrão para volume
        if self.take_profit is None or self.stop_loss is None or self.leverage is None:
            # Calcular ou sugerir valores padrões para take_profit, stop_loss e leverage
            self.take_profit = entry_price * 1.02 if self.take_profit is None else self.take_profit
            self.stop_loss = entry_price * 0.98 if self.stop_loss is None else self.stop_loss
            self.leverage = 10 if self.leverage is None else self.leverage

        risk = self.calculate_risk(entry_price, self.stop_loss, self.volume, self.leverage)
        reward = self.calculate_reward(entry_price, self.take_profit, self.volume, self.leverage)
        ratio = self.risk_reward_ratio(entry_price, self.take_profit, self.stop_loss, self.volume, self.leverage)
        suggestion = self.suggested_trade(ratio)

        new_take_profit, new_stop_loss = self.optimize_take_profit_stop_loss(entry_price, self.stop_loss, self.volume,
                                                                             self.leverage,
                                                                             self.desired_risk_reward_ratio)
        trade_success_probability = self.predict_trade_success(entry_price, self.take_profit, self.stop_loss,
                                                               self.volume, self.leverage)

        # Atualizar o modelo
        self.update_model()

        # Salvar o trade na base de dados (assumindo sucesso ou falha manualmente por agora)
        self.save_trade(entry_price, self.take_profit, self.stop_loss, self.volume, self.leverage,
                        success=1 if trade_success_probability > 0.5 else 0)

        analysis = {
            "Preço de Entrada Atualizado": entry_price,
            "Risco (Perda Máxima)": risk,
            "Recompensa (Ganho Máximo)": reward,
            "Relação Risco/Recompensa": ratio,
            "Sugestão de Trade": suggestion,
            "Novo Take Profit Sugerido": new_take_profit,
            "Novo Stop Loss Sugerido": new_stop_loss,
            "Probabilidade de Sucesso do Trade (IA)": trade_success_probability
        }

        # Previsão de preços futuros
        price_forecast = self.forecast_price()
        analysis["Previsão de Preços Futuros"] = price_forecast

        return analysis


# Exemplo de uso com parâmetros do usuário
symbol = input("Digite o símbolo da criptomoeda (ex: BTC): ").upper()
api_key = input("Digite sua chave da API Alpha Vantage: ")
interval = input("Escolha o intervalo de tempo (ex: 1min, 5min, 15min): ")
take_profit = float(input("Digite o valor de take_profit (ou deixe em branco para IA decidir): ") or 0)
stop_loss = float(input("Digite o valor de stop_loss (ou deixe em branco para IA decidir): ") or 0)
leverage = float(input("Digite o valor de leverage (ou deixe em branco para IA decidir): ") or 0)
volume = float(input("Digite o valor de volume (ou deixe em branco para IA decidir): ") or 0)
desired_risk_reward_ratio = float(input("Digite o valor desejado de relação risco/recompensa (default é 2.0): ") or 2.0)

# Configurações dos parâmetros de trading
take_profit = None if take_profit == 0 else take_profit
stop_loss = None if stop_loss == 0 else stop_loss
leverage = None if leverage == 0 else leverage
volume = None if volume == 0 else volume

analyzer = TradeAnalyzer(symbol, api_key, interval=interval, take_profit=take_profit, stop_loss=stop_loss,
                         leverage=leverage, volume=volume, desired_risk_reward_ratio=desired_risk_reward_ratio)
analysis = analyzer.analyze_trade()

for key, value in analysis.items():
    if key == "Previsão de Preços Futuros":
        print(f"{key}:")
        print(value.tail())  # Mostra as últimas previsões
    else:
        print(f"{key}: {value}")
