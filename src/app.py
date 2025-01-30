import pg8000
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import requests
import time
from datetime import datetime
from colorama import Fore, Style, init
from gpt4all import GPT4All  # Substituindo LlamaCpp por GPT4All

# Inicializa o colorama
init(autoreset=True)

class TradeAnalyzer:
    def __init__(self, symbol, interval, take_profit=None, stop_loss=None, leverage=None, volume=None,
                 desired_risk_reward_ratio=2.0, model=None, cache_duration=60):
        self.symbol = symbol
        self.interval = interval
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.leverage = leverage
        self.volume = volume
        self.desired_risk_reward_ratio = desired_risk_reward_ratio
        self.model = model
        self.cache_duration = cache_duration

        # Inicializa o modelo GPT4All
        try:
            self.llm = GPT4All("mistral-7b-instruct-v0.1.Q4_K_M.gguf")
            self.llm_available = True
            print(f"{Fore.GREEN}Modelo GPT4All carregado com sucesso!")
        except Exception as e:
            print(f"{Fore.YELLOW}Aviso: Não foi possível carregar o modelo GPT4All. Erro: {e}")
            self.llm_available = False

    def predict_with_gpt4all(self, historical_data):
        """Faz previsão usando o modelo GPT4All"""
        if not self.llm_available:
            print(f"{Fore.YELLOW}GPT4All não está disponível. Retornando None.")
            return None

        prompt = f"""
        Analise os últimos preços do {self.symbol}: {historical_data}
        
        Com base nesses dados históricos:
        1. Qual é a tendência atual?
        2. Qual o próximo preço provável?
        3. Qual sua confiança na previsão (0-100%)?
        
        Responda no formato:
        Tendência: [ALTA/BAIXA]
        Preço: [NÚMERO]
        Confiança: [NÚMERO]%
        """

        try:
            response = self.llm.generate(prompt)
            
            lines = response.strip().split('\n')
            prediction_data = {}
            
            for line in lines:
                if 'Preço:' in line:
                    try:
                        prediction_data['price'] = float(''.join(filter(
                            lambda x: x.isdigit() or x == '.', 
                            line.split('Preço:')[1]
                        )))
                    except:
                        prediction_data['price'] = None
                        
                if 'Confiança:' in line:
                    try:
                        prediction_data['confidence'] = float(''.join(filter(
                            lambda x: x.isdigit() or x == '.', 
                            line.split('Confiança:')[1]
                        )))
                    except:
                        prediction_data['confidence'] = 0
                        
                if 'Tendência:' in line:
                    prediction_data['trend'] = 'ALTA' if 'ALTA' in line.upper() else 'BAIXA'
            
            return prediction_data
            
        except Exception as e:
            print(f"{Fore.RED}Erro ao fazer previsão com GPT4All: {e}")
            return None

if __name__ == "__main__":
    print(f"{Fore.CYAN}=== Análise de Investimento em Cripto ===")
    symbol = input(f"{Fore.GREEN}Digite o símbolo da criptomoeda (ex: BTC): ").upper()
    interval = input(f"{Fore.GREEN}Escolha o intervalo de tempo (1m, 5m, 15m, 1h, 4h, 1d): ")
    take_profit = float(input(f"{Fore.GREEN}Take Profit (ex: 1.05 para 5% acima): ") or 0)
    stop_loss = float(input(f"{Fore.GREEN}Stop Loss (ex: 0.95 para 5% abaixo): ") or 0)
    leverage = float(input(f"{Fore.GREEN}Alavancagem (ex: 10 para x10): ") or 1)
    volume = float(input(f"{Fore.GREEN}Volume (quantidade a investir): ") or 100)
    
    analyzer = TradeAnalyzer(symbol, interval, take_profit, stop_loss, leverage, volume)
    historical_data = [100, 102, 101, 105, 107, 106]  # Exemplo fictício de preços passados
    prediction = analyzer.predict_with_gpt4all(historical_data)
    
    if prediction:
        print(f"{Fore.YELLOW}\n=== Resultado da Análise ===")
        print(f"{Fore.WHITE}Tendência: {prediction['trend']}")
        print(f"{Fore.WHITE}Preço Previsto: {prediction['price']}")
        print(f"{Fore.WHITE}Confiança: {prediction['confidence']}%")
    else:
        print(f"{Fore.RED}Não foi possível gerar uma previsão.")
