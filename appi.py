import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
from PIL import Image
import streamlit as st
import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
import streamlit as st

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from scipy.stats import norm
                
 
import plotly.graph_objs as go
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

# --------- Pricing and Simulation Functions ---------
def garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type):
    if T <= 0:
        raise ValueError("La maturité doit être positive")
    d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-r_f*T) * norm.cdf(d1) - K * np.exp(-r_d*T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r_d*T) * norm.cdf(-d2) - S * np.exp(-r_f*T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")



# -------- Fonctions de pricing FX et greeks --------

def garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type):
    if T <= 0:
        raise ValueError("La maturité doit être positive")
    d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S * np.exp(-r_f*T) * norm.cdf(d1) - K * np.exp(-r_d*T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r_d*T) * norm.cdf(-d2) - S * np.exp(-r_f*T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")
    return price

def greeks_garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type):
    if T <= 0:
        raise ValueError("La maturité doit être positive.")
    d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        delta = np.exp(-r_f * T) * norm.cdf(d1)
    elif option_type == 'put':
        delta = -np.exp(-r_f * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")
    gamma = (np.exp(-r_f * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-r_f * T) * norm.pdf(d1) * np.sqrt(T)
    if option_type == 'call':
        theta = (
            - (S * np.exp(-r_f * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r_f * S * np.exp(-r_f * T) * norm.cdf(d1)
            + r_d * K * np.exp(-r_d * T) * norm.cdf(d2)
        ) / 365
    else:
        theta = (
            - (S * np.exp(-r_f * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r_f * S * np.exp(-r_f * T) * norm.cdf(-d1)
            - r_d * K * np.exp(-r_d * T) * norm.cdf(-d2)
        ) / 365
    if option_type == 'call':
        rho_d = T * K * np.exp(-r_d * T) * norm.cdf(d2) / 100
    else:
        rho_d = -T * K * np.exp(-r_d * T) * norm.cdf(-d2) / 100
    if option_type == 'call':
        rho_f = -T * S * np.exp(-r_f * T) * norm.cdf(d1) / 100
    else:
        rho_f = T * S * np.exp(-r_f * T) * norm.cdf(-d1) / 100
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho_d": rho_d,
        "rho_f": rho_f
    }

def binomial_tree_fx(S, K, T, r_d, r_f, sigma, steps, option_type, option_style):
    if steps <= 0:
        raise ValueError("Le nombre d'étapes doit être supérieur à 0.")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r_d - r_f) * dt) - d) / (u - d)
    discount = np.exp(-r_d * dt)
    prices = np.zeros((steps + 1, steps + 1))
    prices[0, 0] = S
    for i in range(1, steps + 1):
        for j in range(i + 1):
            prices[j, i] = S * (u ** (i - j)) * (d ** j)
    values = np.zeros((steps + 1, steps + 1))
    if option_type == 'call':
        values[:, steps] = np.maximum(prices[:, steps] - K, 0)
    elif option_type == 'put':
        values[:, steps] = np.maximum(K - prices[:, steps], 0)
    else:
        raise ValueError("Le type d'option doit être 'call' ou 'put'.")
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = discount * (p * values[j, i + 1] + (1 - p) * values[j + 1, i + 1])
            if option_style == 'american':
                if option_type == 'call':
                    values[j, i] = max(continuation_value, prices[j, i] - K)
                else:
                    values[j, i] = max(continuation_value, K - prices[j, i])
            else:
                values[j, i] = continuation_value
    return prices, values

def trinomial_tree_fx(S, K, T, r_d, r_f, sigma, steps, option_type, option_style):
    if steps <= 0:
        raise ValueError("Le nombre d'étapes doit être supérieur à 0.")
    dt = T / steps
    nu = r_d - r_f
    dx = sigma * np.sqrt(2 * dt)
    u = np.exp(dx)
    d = np.exp(-dx)
    pu = 0.5 * (((sigma**2*dt + (nu*dt)**2)/(dx**2)) + (nu*dt/dx))
    pd = 0.5 * (((sigma**2*dt + (nu*dt)**2)/(dx**2)) - (nu*dt/dx))
    pm = 1 - pu - pd
    if not (0 <= pu <= 1 and 0 <= pm <= 1 and 0 <= pd <= 1):
        raise ValueError("Les probabilités calculées ne sont pas valides. Vérifiez les paramètres.")
    prices = np.zeros((2 * steps + 1, steps + 1))
    prices[steps, 0] = S
    for i in range(1, steps + 1):
        for j in range(-i, i + 1):
            prices[steps + j, i] = S * (u ** max(j, 0)) * (d ** max(-j, 0))
    values = np.zeros((2 * steps + 1, steps + 1))
    if option_type == 'call':
        values[:, steps] = np.maximum(prices[:, steps] - K, 0)
    elif option_type == 'put':
        values[:, steps] = np.maximum(K - prices[:, steps], 0)
    else:
        raise ValueError("Le type d'option doit être 'call' ou 'put'.")
    discount = np.exp(-r_d * dt)
    for i in range(steps - 1, -1, -1):
        for j in range(-i, i + 1):
            idx = steps + j
            continuation_value = discount * (
                pu * values[idx + 1, i + 1] +
                pm * values[idx, i + 1] +
                pd * values[idx - 1, i + 1]
            )
            if option_style == 'american':
                if option_type == 'call':
                    values[idx, i] = max(continuation_value, prices[idx, i] - K)
                elif option_type == 'put':
                    values[idx, i] = max(continuation_value, K - prices[idx, i])
            else:
                values[idx, i] = continuation_value
    return prices, values

def monte_carlo_fx(S, K, T, r_d, r_f, sigma, option_type, simulations=10000, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    Z = np.random.standard_normal(simulations)
    drift = (r_d - r_f - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S * np.exp(drift + diffusion)
    if option_type == 'call':
        payoff = np.maximum(S_T - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")
    price = np.exp(-r_d * T) * np.mean(payoff)
    return price

def draw_tree(prices, values, steps, option_type, model='binomial'):
    """
    Dessine l'arbre des prix et des valeurs pour les modèles binomial ou trinomial (FX compatible).
    """
    G = nx.DiGraph()

    # Crée les nœuds et leurs positions
    if model == 'binomial':
        for i in range(steps + 1):
            for j in range(i + 1):
                G.add_node((j, i), price=prices[j, i], value=values[j, i])
        for i in range(steps):
            for j in range(i + 1):
                G.add_edge((j, i), (j, i + 1))
                G.add_edge((j, i), (j + 1, i + 1))
    elif model == 'trinomial':
        for i in range(steps + 1):
            for j in range(-i, i + 1):
                G.add_node((j, i), price=prices[steps + j, i], value=values[steps + j, i])
        for i in range(steps):
            for j in range(-i, i + 1):
                G.add_edge((j, i), (j + 1, i + 1))
                G.add_edge((j, i), (j, i + 1))
                G.add_edge((j, i), (j - 1, i + 1))
    else:
        raise ValueError("Modèle non pris en charge. Utilisez 'binomial' ou 'trinomial'.")

    # Position des nœuds pour le dessin
    if model == 'binomial':
        pos = {(j, i): (i, -j) for i in range(steps + 1) for j in range(i + 1)}
    elif model == 'trinomial':
        pos = {(j, i): (i, -j) for i in range(steps + 1) for j in range(-i, i + 1)}

    # Labels des nœuds
    labels = {node: f'{G.nodes[node]["price"]:.2f}\n({G.nodes[node]["value"]:.2f})' for node in G.nodes()}

    # Dessiner le graphe avec Matplotlib
    fig, ax = plt.subplots(figsize=(2 + steps, 2 + steps // 2))
    nx.draw(
        G, pos, with_labels=True, labels=labels, node_size=700, node_color="lightblue",
        font_size=8, font_weight="bold", ax=ax
    )

    # Calcul du chemin "optimal" (greedy max)
    path = []
    current = (0, 0)
    for i in range(steps):
        if model == 'binomial':
            v1 = values[current[0], i + 1]
            v2 = values[current[0] + 1, i + 1]
            next_node = (current[0], i + 1) if v1 >= v2 else (current[0] + 1, i + 1)
        elif model == 'trinomial':
            up = values[steps + current[0] + 1, i + 1]
            mid = values[steps + current[0], i + 1]
            down = values[steps + current[0] - 1, i + 1]
            if up >= mid and up >= down:
                next_node = (current[0] + 1, i + 1)
            elif mid >= up and mid >= down:
                next_node = (current[0], i + 1)
            else:
                next_node = (current[0] - 1, i + 1)
        path.append((current, next_node))
        current = next_node

    nx.draw_networkx_edges(G, pos, edgelist=path, edge_color="red", width=2.5, ax=ax)
    plt.title(f'{model.capitalize()} Tree ({option_type.capitalize()} Option)')
    plt.tight_layout()

    return fig  # Retourne la figure

def plot_fx_option_price_vs_params(
    S, K, T, r_d, r_f, sigma, option_type='call', param_to_vary='S', param_range=None, steps=100
):
    """
    Trace le prix de l'option FX en fonction d'un paramètre variable.

    :param S: Spot FX rate.
    :param K: Strike price.
    :param T: Temps à maturité (en années).
    :param r_d: Taux d'intérêt domestique.
    :param r_f: Taux d'intérêt étranger.
    :param sigma: Volatilité.
    :param option_type: Type d'option ('call' ou 'put').
    :param param_to_vary: Le paramètre à faire varier ('S', 'K', 'T', 'r_d', 'r_f', 'sigma').
    :param param_range: Plage de valeurs pour le paramètre à varier (par défaut, basé sur des valeurs raisonnables).
    :param steps: Nombre de points à tracer.
    """
    if param_range is None:
        if param_to_vary == 'S':
            param_range = np.linspace(S * 0.5, S * 1.5, steps)
        elif param_to_vary == 'K':
            param_range = np.linspace(K * 0.5, K * 1.5, steps)
        elif param_to_vary == 'T':
            param_range = np.linspace(0.01, T * 2, steps)
        elif param_to_vary == 'r_d':
            param_range = np.linspace(r_d - 0.05, r_d + 0.05, steps)
        elif param_to_vary == 'r_f':
            param_range = np.linspace(r_f - 0.05, r_f + 0.05, steps)
        elif param_to_vary == 'sigma':
            param_range = np.linspace(sigma * 0.5, sigma * 1.5, steps)
        else:
            raise ValueError(f"Paramètre non pris en charge : {param_to_vary}")

    prices = []
    for param_value in param_range:
        if param_to_vary == 'S':
            price = garman_kohlhagen(param_value, K, T, r_d, r_f, sigma, option_type)
        elif param_to_vary == 'K':
            price = garman_kohlhagen(S, param_value, T, r_d, r_f, sigma, option_type)
        elif param_to_vary == 'T':
            price = garman_kohlhagen(S, K, param_value, r_d, r_f, sigma, option_type)
        elif param_to_vary == 'r_d':
            price = garman_kohlhagen(S, K, T, param_value, r_f, sigma, option_type)
        elif param_to_vary == 'r_f':
            price = garman_kohlhagen(S, K, T, r_d, param_value, sigma, option_type)
        elif param_to_vary == 'sigma':
            price = garman_kohlhagen(S, K, T, r_d, r_f, param_value, option_type)
        else:
            raise ValueError(f"Paramètre non pris en charge : {param_to_vary}")
        prices.append(price)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, prices, label=f"Prix de l'option ({option_type}) en fonction de {param_to_vary}")
    plt.xlabel(param_to_vary)
    plt.ylabel("Prix de l'option")
    plt.title(f"Impact de {param_to_vary} sur le prix de l'option ({option_type.capitalize()})")
    plt.legend()
    plt.grid()
    plt.show()
def plot_convergence(S, K, T, r_d, r_f, sigma, option_type, option_style):
    steps_range = range(2, 51)
    binomial_prices = []
    trinomial_prices = []
    mc_prices = []
    for steps in steps_range:
        _, values_bi = binomial_tree_fx(S, K, T, r_d, r_f, sigma, steps, option_type, option_style)
        binomial_prices.append(values_bi[0, 0])
        try:
            _, values_tri = trinomial_tree_fx(S, K, T, r_d, r_f, sigma, steps, option_type, option_style)
            trinomial_prices.append(values_tri[steps, 0])
        except:
            trinomial_prices.append(np.nan)
        prix_mc = monte_carlo_fx(S, K, T, r_d, r_f, sigma, option_type, simulations=1000*steps, random_seed=42)
        mc_prices.append(prix_mc)
    price_gk = garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type)
    fig, ax = plt.subplots(figsize=(6, 4))  # Taille modifiée ici aussi
    ax.plot(steps_range, binomial_prices, label="Binomial FX")
    ax.plot(steps_range, trinomial_prices, label="Trinomial FX")
    ax.plot(steps_range, mc_prices, label="Monte Carlo FX", alpha=0.7)
    ax.hlines(price_gk, steps_range.start, steps_range.stop-1, colors='k', linestyles='dashed', label="Garman-Kohlhagen (analytique)")
    ax.set_xlabel("Nombre de pas (ou x1000 simulations MC)")
    ax.set_ylabel("Prix de l'option")
    ax.set_title("Convergence des modèles FX vers le prix analytique")
    ax.legend()
    ax.grid()
    st.pyplot(fig)


# ============================ MODELE BINOMIAL ============================

def binomial_tree(S, K, T, r, sigma, steps, option_type, option_style):
    """
    Calcule le prix d'une option via le modèle binomial.

    Args:
        S (float): Prix initial de l'actif sous-jacent.
        K (float): Prix d'exercice.
        T (float): Temps jusqu'à l'expiration (en années).
        r (float): Taux sans risque.
        sigma (float): Volatilité.
        steps (int): Nombre d'étapes dans l'arbre binomial.
        option_type (str): Type d'option ('call' ou 'put').
        option_style (str): Style d'option ('european' ou 'american').

    Returns:
        tuple: Matrices des prix et des valeurs de l'arbre binomial.
    """
    if steps <= 0:
        raise ValueError("Le nombre d'étapes doit être supérieur à 0.")
    
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Initialisation des matrices
    prices = np.zeros((steps + 1, steps + 1))
    prices[0, 0] = S
    for i in range(1, steps + 1):
        for j in range(i + 1):
            prices[j, i] = S * (u ** (i - j)) * (d ** j)

    values = np.zeros((steps + 1, steps + 1))
    if option_type == 'call':
        values[:, steps] = np.maximum(prices[:, steps] - K, 0)
    elif option_type == 'put':
        values[:, steps] = np.maximum(K - prices[:, steps], 0)
    else:
        raise ValueError("Le type d'option doit être 'call' ou 'put'.")

    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = discount * (p * values[j, i + 1] + (1 - p) * values[j + 1, i + 1])
            if option_style == 'american':
                if option_type == 'call':
                    values[j, i] = max(continuation_value, prices[j, i] - K)
                else:
                    values[j, i] = max(continuation_value, K - prices[j, i])
            else:
                values[j, i] = continuation_value

    return prices, values


# ============================ MODELE TRINOMIAL ============================

import numpy as np

def trinomial_tree(S, K, T, r, sigma, steps, option_type, option_style):
    """
    Calcule le prix d'une option via le modèle trinomial, corrigé pour respecter la neutralité au risque.

    Args:
        S (float): Prix initial de l'actif sous-jacent.
        K (float): Prix d'exercice.
        T (float): Temps jusqu'à l'expiration (en années).
        r (float): Taux sans risque.
        sigma (float): Volatilité.
        steps (int): Nombre d'étapes dans l'arbre trinomial.
        option_type (str): Type d'option ('call' ou 'put').
        option_style (str): Style d'option ('european' ou 'american').

    Returns:
        tuple: Matrices des prix et des valeurs de l'arbre trinomial.
    """
    if steps <= 0:
        raise ValueError("Le nombre d'étapes doit être supérieur à 0.")
    
    # Paramètres de temps
    dt = T / steps  # Durée d'une période
    u = np.exp(sigma * np.sqrt(2 * dt))  # Facteur de hausse
    d = 1 / u  # Facteur de baisse
    m = 1  # Facteur stable (aucune variation)

    # Probabilités neutres au risque (pu, pm, pd)
    pu = ((np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt) / 2)) /
          (np.exp(sigma * np.sqrt(dt) / 2) - np.exp(-sigma * np.sqrt(dt) / 2))) ** 2
    pd = ((np.exp(sigma * np.sqrt(dt) / 2) - np.exp(r * dt / 2)) /
          (np.exp(sigma * np.sqrt(dt) / 2) - np.exp(-sigma * np.sqrt(dt) / 2))) ** 2
    pm = 1 - pu - pd

    # Vérification des probabilités
    if not (0 <= pu <= 1 and 0 <= pd <= 1 and 0 <= pm <= 1):
        raise ValueError("Les probabilités calculées ne sont pas valides. Vérifiez les paramètres.")

    # Initialisation des matrices pour les prix et les valeurs
    prices = np.zeros((2 * steps + 1, steps + 1))
    prices[steps, 0] = S  # Prix initial

    # Construction de l'arbre des prix
    for i in range(1, steps + 1):
        for j in range(-i, i + 1):
            prices[steps + j, i] = S * (u ** max(j, 0)) * (d ** max(-j, 0))

    # Initialisation des valeurs des options à la maturité
    values = np.zeros((2 * steps + 1, steps + 1))
    if option_type == 'call':
        values[:, steps] = np.maximum(prices[:, steps] - K, 0)
    elif option_type == 'put':
        values[:, steps] = np.maximum(K - prices[:, steps], 0)
    else:
        raise ValueError("Le type d'option doit être 'call' ou 'put'.")

    # Backward induction pour calculer les valeurs
    discount = np.exp(-r * dt)  # Facteur d'actualisation
    for i in range(steps - 1, -1, -1):
        for j in range(-i, i + 1):
            continuation_value = discount * (
                pu * values[steps + j + 1, i + 1] +
                pm * values[steps + j, i + 1] +
                pd * values[steps + j - 1, i + 1]
            )
            # Si option américaine, vérifier l'exercice anticipé
            if option_style == 'american':
                if option_type == 'call':
                    values[steps + j, i] = max(continuation_value, prices[steps + j, i] - K)
                elif option_type == 'put':
                    values[steps + j, i] = max(continuation_value, K - prices[steps + j, i])
            else:  # Option européenne
                values[steps + j, i] = continuation_value

    return prices, values

# ============================ MONTE CARLO ============================

def monte_carlo_pricing(S, K, T, r, sigma, option_type, num_simulations=10000):
    """
    Calcule le prix d'une option via la simulation Monte Carlo.

    Args:
        S (float): Prix initial de l'actif sous-jacent.
        K (float): Prix d'exercice.
        T (float): Temps jusqu'à l'expiration (en années).
        r (float): Taux sans risque.
        sigma (float): Volatilité.
        option_type (str): Type d'option ('call' ou 'put').
        num_simulations (int): Nombre de simulations Monte Carlo.

    Returns:
        float: Prix estimé de l'option.
    """
    if num_simulations <= 0:
        raise ValueError("Le nombre de simulations doit être supérieur à 0.")
    
    dt = T
    payoff = np.zeros(num_simulations)

    for i in range(num_simulations):
        ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        if option_type == 'call':
            payoff[i] = max(0, ST - K)
        elif option_type == 'put':
            payoff[i] = max(0, K - ST)
        else:
            raise ValueError("Le type d'option doit être 'call' ou 'put'.")

    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price


# ============================ BLACK-SCHOLES ============================

def black_scholes(S, K, T, r, sigma, option_type):
    """
    Calcule le prix d'une option via le modèle Black-Scholes.

    Args:
        S (float): Prix initial de l'actif sous-jacent.
        K (float): Prix d'exercice.
        T (float): Temps jusqu'à l'expiration (en années).
        r (float): Taux sans risque.
        sigma (float): Volatilité.
        option_type (str): Type d'option ('call' ou 'put').

    Returns:
        float: Prix de l'option.
    """
    if T <= 0:
        raise ValueError("La maturité (T) doit être strictement positive.")
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Le type d'option doit être 'call' ou 'put'.")
    
    return price


# ============================ CALCUL DES GRECS ============================

def calculate_greeks(S, K, T, r, sigma, option_type):
    """
    Calcule les grecs principaux (Delta, Gamma, Theta, Vega, Rho).

    Args:
        S (float): Prix initial de l'actif sous-jacent.
        K (float): Prix d'exercice.
        T (float): Temps jusqu'à l'expiration (en années).
        r (float): Taux sans risque.
        sigma (float): Volatilité.
        option_type (str): Type d'option ('call' ou 'put').

    Returns:
        tuple: Delta, Gamma, Theta, Vega, Rho.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = - K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

    return delta, gamma, theta, vega, rho




# ============================ TRAÇAGE D'ÉVOLUTION ============================

def plot_evolution(S, K, T, r, sigma, steps, num_simulations):
    """
    Trace l'évolution des prix d'options pour différents modèles.

    Args:
        S (float): Prix initial de l'actif sous-jacent.
        K (float): Prix d'exercice.
        T (float): Temps jusqu'à l'expiration (en années).
        r (float): Taux sans risque.
        sigma (float): Volatilité.
        steps (int): Nombre d'étapes dans les arbres binomial/trinomial.
        num_simulations (int): Nombre de simulations Monte Carlo.

    Returns:
        matplotlib.figure.Figure: La figure contenant le graphique.
    """
    # Création explicite de la figure et des axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Exclure une maturité de 0 en commençant légèrement au-dessus de 0
    time_to_maturity = np.linspace(0.01, T, 100)

    # Simulation des prix des options avec différentes méthodes
    prices_bs = [black_scholes(S, K, t, r, sigma, 'call') for t in time_to_maturity]
    prices_mc = [monte_carlo_pricing(S, K, t, r, sigma, 'call', num_simulations) for t in time_to_maturity]
    
    # Calcul des prix pour les modèles binomial et trinomial
    prices_binomial = []
    prices_trinomial = []
    
    for t in time_to_maturity:
        # Binomial Tree
        _, values_bin = binomial_tree(S, K, t, r, sigma, steps, 'call', 'european')
        prices_binomial.append(values_bin[0, 0])
        
        # Trinomial Tree
        _, values_tri = trinomial_tree(S, K, t, r, sigma, steps, 'call', 'european')
        prices_trinomial.append(values_tri[steps, 0])

    # Tracer toutes les courbes sur le même axe
    ax.plot(time_to_maturity, prices_bs, label='Black-Scholes', color='blue', lw=2)
    ax.plot(time_to_maturity, prices_mc, label='Monte Carlo', color='orange')
    ax.plot(time_to_maturity, prices_binomial, label='Binomial Tree', color='green')
    ax.plot(time_to_maturity, prices_trinomial, label='Trinomial Tree', color='red')
    
    # Personnalisation du graphique
    ax.set_title('Évolution des prix des options en fonction de la maturité', fontsize=14)
    ax.set_xlabel('Maturité (années)', fontsize=12)
    ax.set_ylabel('Prix de l\'option', fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Amélioration de l'espacement
    fig.tight_layout()
    
    return fig  # Retourne explicitement la figure

# ============================ DRAW TREE ============================

import networkx as nx
import matplotlib.pyplot as plt

def draw_treee(prices, values, steps, option_type, model='binomial'):
    """
    Dessine l'arbre des prix et des valeurs pour les modèles binomial ou trinomial.

    Args:
        prices (np.ndarray): Matrice des prix des actifs aux nœuds.
        values (np.ndarray): Matrice des valeurs des options aux nœuds.
        steps (int): Nombre d'étapes dans l'arbre.
        option_type (str): Type d'option ('call' ou 'put').
        model (str): Modèle utilisé ('binomial' ou 'trinomial').

    Returns:
        None: Affiche directement le graphe.
    """
    G = nx.DiGraph()

    # Crée les nœuds et leurs positions
    if model == 'binomial':
        for i in range(steps + 1):
            for j in range(i + 1):
                G.add_node((j, i), price=prices[j, i], value=values[j, i])
        for i in range(steps):
            for j in range(i + 1):
                G.add_edge((j, i), (j, i + 1))
                G.add_edge((j, i), (j + 1, i + 1))
    elif model == 'trinomial':
        for i in range(steps + 1):
            for j in range(-i, i + 1):
                G.add_node((j, i), price=prices[steps + j, i], value=values[steps + j, i])
        for i in range(steps):
            for j in range(-i, i + 1):
                G.add_edge((j, i), (j + 1, i + 1))
                G.add_edge((j, i), (j, i + 1))
                G.add_edge((j, i), (j - 1, i + 1))
    else:
        raise ValueError("Modèle non pris en charge. Utilisez 'binomial' ou 'trinomial'.")

    # Position des nœuds
    pos = {}
    if model == 'binomial':
        pos = {(j, i): (i, -j) for i in range(steps + 1) for j in range(i + 1)}
    elif model == 'trinomial':
        pos = {(j, i): (i, -j) for i in range(steps + 1) for j in range(-i, i + 1)}

    # Labels des nœuds
    labels = {node: f'{G.nodes[node]["price"]:.2f}\n({G.nodes[node]["value"]:.2f})' for node in G.nodes()}

    # Dessiner le graphe
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, node_color="lightblue", font_size=8, font_weight="bold")

    # Calcul du chemin optimal
    path = []
    current = (0, 0) if model == 'binomial' else (0, 0)  # Racine de l'arbre
    for i in range(steps):
        if model == 'binomial':
            # Choisir le nœud avec la plus grande valeur parmi les enfants directs
            next_node = (current[0], i + 1) if values[current[0], i + 1] >= values[current[0] + 1, i + 1] else (current[0] + 1, i + 1)
        elif model == 'trinomial':
            # Choisir le nœud avec la plus grande valeur parmi les trois enfants directs
            up = values[steps + current[0] + 1, i + 1]
            mid = values[steps + current[0], i + 1]
            down = values[steps + current[0] - 1, i + 1]
            if up >= mid and up >= down:
                next_node = (current[0] + 1, i + 1)
            elif mid >= up and mid >= down:
                next_node = (current[0], i + 1)
            else:
                next_node = (current[0] - 1, i + 1)
        path.append((current, next_node))
        current = next_node

    # Mettre en évidence le chemin optimal
    nx.draw_networkx_edges(G, pos, edgelist=path, edge_color="red", width=2.5)
    plt.title(f'{model.capitalize()} Tree ({option_type.capitalize()} Option)')
    plt.show()

# ======= CSS FUTURISTE =======
st.markdown("""
    <style>
    /* Fond général sombre avec dégradé fluide */
    body {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Encadrement néon pour l'introduction */
    .intro-box {
        background: rgba(20, 20, 30, 0.7);
        border: 2px solid rgba(0, 255, 255, 0.4);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 0 20px rgba(0,255,255,0.3), 0 0 40px rgba(0,255,255,0.2);
        backdrop-filter: blur(10px);
        color: white;
        font-size: 17px;
        line-height: 1.6;
    }

    /* Titre avec effet lumineux ultra-intense */
    h1 {
        font-size: 4.5em;
        font-weight: bold;
        text-align: center;
        color: #00ffe1;
        text-shadow: 0 0 20px #00ffe1, 0 0 30px #00ffe1, 0 0 50px #00ffe1, 0 0 80px #00ffe1, 0 0 120px #00ffe1, 0 0 180px #00ffe1, 0 0 200px #00ffe1;
        font-family: 'Segoe UI', sans-serif;
        margin-bottom: 20px;
        animation: neonGlow 1s ease-in-out infinite alternate;
    }

    /* Style des onglets (UX moderne) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        padding: 10px;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 18px;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #00ffe1;
        color: #111;
        transform: scale(1.05);
    }

    .stTabs [aria-selected="true"] {
        background-color: #00ffe1;
        color: #111;
        text-shadow: 0 0 30px #00ffe1, 0 0 50px #00ffe1, 0 0 80px #00ffe1, 0 0 120px #00ffe1, 0 0 200px #00ffe1;
        animation: neonGlow 0.8s ease-in-out infinite alternate;
        transform: scale(1.1);
    }

    /* Animation néon pour un effet plus lumineux et dynamique */
    @keyframes neonGlow {
        0% {
            text-shadow: 0 0 20px #00ffe1, 0 0 30px #00ffe1, 0 0 50px #00ffe1, 0 0 80px #00ffe1, 0 0 120px #00ffe1;
        }
        50% {
            text-shadow: 0 0 50px #00ffe1, 0 0 80px #00ffe1, 0 0 120px #00ffe1, 0 0 180px #00ffe1, 0 0 240px #00ffe1;
        }
        100% {
            text-shadow: 0 0 80px #00ffe1, 0 0 120px #00ffe1, 0 0 160px #00ffe1, 0 0 200px #00ffe1, 0 0 300px #00ffe1;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ====== TITRE ======
st.markdown("<h1>💱 Forex Pricing Hedging Platform</h1>", unsafe_allow_html=True)

# ====== INTRO ENCADRÉE ======
intro_html = """
<div class="intro-box">
    <p>Bienvenue sur <strong>Forex Pricing Hedging Platform</strong> 🌐, une plateforme complète pour la gestion des risques de change et la couverture 🛡️.</p>
    <p>La couverture contre le risque de change est essentielle pour atténuer les effets des fluctuations des taux de change sur les activités économiques.</p>
    <p>Dans le contexte de <strong>la préparation à la convertibilité totale du dirham marocain (MAD)</strong>, cette gestion devient encore plus cruciale.</p>
    <p>Les instruments financiers tels que les options de change, les contrats à terme et les swaps sont des outils puissants pour assurer une couverture efficace et garantir la stabilité financière dans ce nouvel environnement économique. D'où l'intérêt de cette plateforme 🌍.</p>
</div>
"""
st.markdown(intro_html, unsafe_allow_html=True)



# Inject CSS
st.markdown("""
    <style>
    /* Sélectionne les onglets (boutons) */
    div[data-baseweb="tab"] button {
        background-color: #00FFFF !important; /* Bleu électrique */
        color: white !important;
        border-radius: 25px !important; /* Rond */
        padding: 8px 20px !important;
        margin-right: 6px !important;
        font-weight: 600;
        border: none !important;
        transition: box-shadow 0.2s;
        box-shadow: 0 0 0px 2px #00FFFF33;
    }
    div[data-baseweb="tab"] button[aria-selected="true"] {
        background-color: #0099FF !important; /* Variante pour l'onglet actif */
        color: white !important;
        box-shadow: 0 0 8px 3px #0099FF55;
    }
    </style>
""", unsafe_allow_html=True)

# Vos onglets
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,tab9, tab10 ,tab11 = st.tabs(
    ["🏠 Accueil", 
     "Economic Calendar 📊", 
     "MarketPulse 📡",
     "FX Forward Pricer 💹", "Pricing FX-Options 💱",
     "Etude de FX-volatilité ⚡","Options Strategy 💼","FX BOOK 📘-Tunnels","FX BOOK 📘-Options Participatives",
     "⚖️ Pricing Swaps",
     "📉 Stress Tests FX"])
import streamlit as st
import investpy
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
# ===== CONTENU DES ONGLETS =====
with tab3:
    st.title("# MarketPulse : FX-DashBoard")
    # ==================== STYLE & INTRO PEDAGOGIQUE ====================
    st.markdown("""
<style>
.pulse-box {
    background: rgba(15, 30, 60, 0.88);
    border: 2px solid #00ffff77;
    border-radius: 18px;
    padding: 18px 26px;
    margin-bottom: 22px;
    margin-top: 10px;
    box-shadow: 0 0 18px 2px #00ffff33, 0 0 35px 10px #00ffff22 inset;
    color: white;
    font-size: 16px;
    line-height: 1.55;
}
.metrique-pulse {
    background: linear-gradient(135deg, #1e1e2f, #29293d);
    padding: 10px 12px;
    border-radius: 10px;
    box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.18);
    color: white;
    font-family: 'Arial', sans-serif;
    margin-bottom: 10px;
    text-align: left !important;
}
.metrique-title {
    font-size: 1rem;
    font-weight: bold;
    color: #FFD700;
}
.metrique-value {
    font-size: 1.5rem;
    font-weight: bold;
}
th, td {
    font-size: 14px !important;
    text-align: left !important;
}
.synthese-pulse {
    background: #eaffd0;
    border-radius: 10px;
    color: #2d4a2d;
    font-size: 16px;
    padding: 12px;
    margin: 8px 0 16px 0;
    border-left: 5px solid #57bf34;
}
.alerte-pulse {
    background: #ffe9e9;
    border-radius: 10px;
    color: #a80000;
    font-size: 15px;
    padding: 10px 12px;
    margin: 10px 0 15px 0;
    border-left: 5px solid #e51515;
}
</style>
<div class="pulse-box">
<b>Bienvenue sur <span style="color:#00ffe2;">MarketPulse : FX-DashBoard</span> !</b><br>
<ul>
    <li>Suivez les <b>taux de change en temps réel</b> sur les principales devises</li>
    <li>Visualisez l’évolution historique sur la période de votre choix</li>
    <li>Analysez les variations, la volatilité et d’autres indicateurs clés</li>
    <li>Exportez les historiques, comparez plusieurs paires et recevez des alertes visuelles</li>
</ul>
<b>À quoi sert ce dashboard ?</b><br>
Il permet d’anticiper les mouvements du marché, d’optimiser vos décisions de trading ou de couverture, et de renforcer votre veille sur les risques de change.
</div>
""", unsafe_allow_html=True)

    # Titre principal de l'application
   
    
    st.markdown("### Données en temps réel, graphiques interactifs et indicateurs clés")

    # ===== Clé d'accès API =====
    API_ACCESS_KEY = "dff368446f2002d3b7c0b83b642bd7bd"

    # ===== Fonction pour obtenir les taux en temps réel =====
    @st.cache_data(ttl=60)
    def get_live_rates():
        url = f"https://api.exchangerate.host/live?access_key={API_ACCESS_KEY}&currencies=MAD,EUR,USD"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    quotes = data["quotes"]
                    rates = {
                        "EUR/MAD": quotes["USDMAD"] / quotes["USDEUR"],
                        "USD/MAD": quotes["USDMAD"],
                        "EUR/USD": quotes["USDEUR"]
                    }
                    return rates
                else:
                    st.error("Erreur API : Données indisponibles.")
                    return {}
            else:
                st.error(f"Erreur HTTP : {response.status_code}")
                return {}
        except Exception as e:
            st.error(f"Erreur technique : {str(e)}")
            return {}

    # ===== Fonction pour obtenir des données historiques =====
    @st.cache_data(ttl=3600)
    def get_historical_data(pair, start_date, end_date):
        base, target = pair.split("/")
        url = f"https://api.exchangerate.host/timeframe?access_key={API_ACCESS_KEY}"
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "currencies": target,
            "source": base
        }
        try:
            # Pause pour respecter les limites de l'API
            time.sleep(1)
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and "quotes" in data:
                    rates = data["quotes"]
                    df = pd.DataFrame([
                        {"Date": date, "Taux": rates[date][f"{base}{target}"]}
                        for date in rates
                    ])
                    df["Date"] = pd.to_datetime(df["Date"])
                    return df
                else:
                    st.warning("Données historiques non disponibles.")
                    return pd.DataFrame()
            else:
                st.error(f"Erreur HTTP : {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Erreur technique historique : {str(e)}")
            return pd.DataFrame()

    # ===== Fonction pour calculer des informations financières =====
    def calculate_financial_info(data):
        if data.empty:
            return {}
        info = {
            "Clôture précédente": round(data["Taux"].iloc[-2], 4) if len(data) > 1 else None,
            "Ouverture": round(data["Taux"].iloc[0], 4),
            "Achat": round(data["Taux"].mean() * 0.998, 4),
            "Vente": round(data["Taux"].mean() * 1.002, 4),
            "Écart journalier": round(data["Taux"].max() - data["Taux"].min(), 4),
            "Variation sur 1 an": f"{round((data['Taux'].iloc[-1] - data['Taux'].iloc[0]) / data['Taux'].iloc[0] * 100, 2)}%"
        }
        return info

    # ===== Données en temps réel =====
    live_rates = get_live_rates()

    if live_rates:
        st.success("✅ Données récupérées avec succès !")
        rates_df = pd.DataFrame(list(live_rates.items()), columns=["Pair", "Taux"])
        st.dataframe(rates_df.style.format(precision=4).background_gradient(cmap="Blues"), hide_index=True)

        # Liste des paires
        pairs_to_display = ["EUR/MAD", "USD/MAD", "EUR/USD"]

        # Colonnes pour afficher les graphiques et les informations
        col1, col2, col3 = st.columns(3)

        for idx, pair in enumerate(pairs_to_display):
            # Sélection de la colonne
            if idx == 0:
                col = col1
            elif idx == 1:
                col = col2
            else:
                col = col3

            with col:
                st.markdown(f"### {pair}")
                start_date = st.date_input(f"Date de début ({pair})", value=datetime.today() - timedelta(days=30), key=f"start_{pair}")
                end_date = st.date_input(f"Date de fin ({pair})", value=datetime.today(), key=f"end_{pair}")

                if start_date and end_date and start_date <= end_date:
                    historical_data = get_historical_data(pair, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    if not historical_data.empty:
                        # Graphique interactif
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=historical_data["Date"],
                            y=historical_data["Taux"],
                            mode="lines",
                            name="Taux"
                        ))
                        fig.update_layout(
                            title=f"Évolution de {pair}",
                            xaxis_title="Date",
                            yaxis_title="Taux",
                            template="plotly_white",
                            xaxis_rangeslider_visible=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Informations financières dans un tableau stylisé
                        st.markdown(f"#### Informations financières pour {pair}")
                        financial_info = calculate_financial_info(historical_data)
                        financial_df = pd.DataFrame(list(financial_info.items()), columns=["Indicateur", "Valeur"])
                        st.dataframe(financial_df.style.set_properties(**{'text-align': 'left'}).background_gradient(cmap="Greens"), hide_index=True)
                    else:
                        st.warning(f"Aucune donnée historique trouvée pour {pair}.")
    else:
        st.warning("Impossible de récupérer les taux de change.")
        
    # ==================== SOURCES & DISCLAIMER ====================
    st.markdown("""
---
<span style="font-size:13px;color:#888;">
Données fournies à titre indicatif, issues de l’API <a href="https://exchangerate.host" target="_blank">exchangerate.host</a>.<br>
<b>API Key :</b> Gérée via variable d’environnement <code>FX_API_KEY</code> (clé de test utilisée si absente).
</span>
""", unsafe_allow_html=True)    
        
with tab2:
    import streamlit as st
    import investpy
    import pandas as pd
    from datetime import datetime, timedelta
    st.title("### 📆 Calendrier Économique")

    # ------ Encadré d'utilité pédagogique ------
    st.markdown("""
    <style>
    .utilite-box {
        background: rgba(10, 18, 35, 0.88);
        border: 2px solid #00ffff77;
        border-radius: 18px;
        padding: 16px 24px;
        margin-bottom: 22px;
        margin-top: 10px;
        box-shadow: 0 0 18px 2px #00ffff33, 0 0 35px 10px #00ffff22 inset;
        color: white;
        font-size: 16px;
        line-height: 1.5;
    }
    .intro-box {
        background: rgba(20, 20, 30, 0.8);
        border: 2px solid rgba(0, 255, 255, 0.4);
        border-radius: 20px;
        padding: 20px 22px;
        margin-bottom: 28px;
        box-shadow: 0 0 18px 2px #00ffff44, 0 0 30px 8px #00ffff22 inset;
        color: white;
        font-size: 15px;
        line-height: 1.5;
    }
    .metric-box {
        background: linear-gradient(135deg, #1e1e2f, #29293d);
        padding: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.18);
        color: white;
        font-family: 'Arial', sans-serif;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 1rem;
        font-weight: bold;
        color: #FFD700;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    th, td {
        font-size: 14px !important;
        text-align: left !important;
    }
    </style>
    <div class="utilite-box">
    <b>Ce module vous permet de suivre et d’analyser les événements macroéconomiques majeurs à venir.</b>
    <ul>
        <li>Filtrez les annonces par pays, importance et période</li>
        <li>Identifiez les publications clés (banques centrales, inflation, emploi…)</li>
        <li>Anticipez les impacts potentiels sur les marchés financiers</li>
        <li>Exportez ou recherchez facilement vos événements</li>
        <li>Optimisez votre veille et la gestion du risque macroéconomique</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # ------ Checkbox pour afficher la description détaillée ------
    show_desc = st.checkbox("Afficher les détails pédagogiques sur le calendrier économique")

    if show_desc:
        st.markdown("""
        <div class="intro-box">
        <h4>📅 Pourquoi utiliser un calendrier économique ?</h4>
        <ul>
            <li>Les marchés réagissent fortement aux publications économiques, surtout si l’annonce diffère du consensus attendu.</li>
            <li>Un calendrier personnalisé permet d’anticiper la volatilité et de planifier vos décisions d’investissement ou de couverture.</li>
            <li>Filtrer par importance et pays vous aide à vous concentrer sur les événements qui vous concernent réellement.</li>
        </ul>
        <b>Exemples d’événements :</b> taux directeurs, inflation (CPI), chômage, PIB, balance commerciale, discours de banquiers centraux, etc.<br>
        <b>Astuce :</b> Les annonces “High” importance sont souvent les plus susceptibles de déclencher de la volatilité sur les actifs.</div>
        """, unsafe_allow_html=True)

    # ------ Titre principal du module ------
   
    # ------ Mapping pays ------
    country_mapping = {
        'Maroc': 'morocco',
        'Zone Euro': 'euro area',
        'Royaume-Uni': 'united kingdom',
        'États-Unis': 'united states',
        'France': 'france',
        'Allemagne': 'germany',
        'Japon': 'japan',
        'Chine': 'china',
        'Canada': 'canada',
        'Suisse': 'switzerland',
        'Australie': 'australia'
    }

    # ------ Sélecteurs ------
    col1, col2, col3 = st.columns([2,1,1])

    with col1:
        selected_countries = st.multiselect(
            "Pays",
            options=list(country_mapping.keys()),
            default=['États-Unis', 'Zone Euro', 'France', 'Allemagne']
        )

    with col2:
        importance_levels = st.multiselect(
            "Niveau d'importance",
            options=['High', 'Medium', 'Low'],
            default=['High', 'Medium']
        )

    with col3:
        date_range = st.date_input(
            "Période",
            value=[
                datetime.today().date(),
                (datetime.today() + timedelta(days=7)).date()
            ],
            min_value=datetime.today().date(),
            max_value=(datetime.today() + timedelta(days=30)).date()
        )

    # ------ Récupération des données ------
    def fetch_economic_calendar(countries, importances, from_date, to_date):
        try:
            df = investpy.economic_calendar(
                countries=countries,
                from_date=from_date.strftime('%d/%m/%Y'),
                to_date=to_date.strftime('%d/%m/%Y'),
                importances=importances
            )
            if not df.empty:
                # Correction ici : on ne renomme que les colonnes existantes
                rename_dict = {}
                if 'importance' in df.columns: rename_dict['importance'] = 'Importance'
                if 'event' in df.columns: rename_dict['event'] = 'Événement'
                if 'country' in df.columns: rename_dict['country'] = 'Pays'
                if 'date' in df.columns: rename_dict['date'] = 'Date'
                if 'time' in df.columns: rename_dict['time'] = 'Heure'
                if 'zone' in df.columns: rename_dict['zone'] = 'Zone'
                df = df.rename(columns=rename_dict)
                if 'Importance' in df.columns:
                    df['Importance'] = df['Importance'].str.capitalize()
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
                if 'Heure' not in df.columns:
                    df['Heure'] = ''
                return df.sort_values(by=[col for col in ['Date','Heure'] if col in df.columns])
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Erreur technique : {str(e)}")
            return pd.DataFrame()

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        from_date, to_date = date_range
        if from_date > to_date:
            st.warning("Veuillez sélectionner une plage de dates valide.")
        elif not selected_countries:
            st.warning("Veuillez sélectionner au moins un pays.")
        else:
            countries = [country_mapping[c] for c in selected_countries]
            with st.spinner("Chargement du calendrier économique en cours ..."):
                df = fetch_economic_calendar(
                    countries=countries,
                    importances=importance_levels,
                    from_date=from_date,
                    to_date=to_date
                )
            # ------ Compteurs et résumé ------
            if not df.empty:
                n_total = len(df)
                n_high = (df['Importance'] == 'High').sum() if 'Importance' in df.columns else 0
                n_med = (df['Importance'] == 'Medium').sum() if 'Importance' in df.columns else 0
                n_low = (df['Importance'] == 'Low').sum() if 'Importance' in df.columns else 0

                metric_cols = st.columns(4)
                metric_cols[0].markdown(f"<div class='metric-box'><div class='metric-title'>Événements</div><div class='metric-value'>{n_total}</div></div>", unsafe_allow_html=True)
                metric_cols[1].markdown(f"<div class='metric-box'><div class='metric-title'>High</div><div class='metric-value'>{n_high}</div></div>", unsafe_allow_html=True)
                metric_cols[2].markdown(f"<div class='metric-box'><div class='metric-title'>Medium</div><div class='metric-value'>{n_med}</div></div>", unsafe_allow_html=True)
                metric_cols[3].markdown(f"<div class='metric-box'><div class='metric-title'>Low</div><div class='metric-value'>{n_low}</div></div>", unsafe_allow_html=True)

                # ------ Events par pays ------
                if 'Pays' in df.columns:
                    n_country = df['Pays'].value_counts().to_dict()
                    pays_count_str = " | ".join([f"{k}: {v}" for k,v in n_country.items()])
                    st.markdown(f"<span style='color:#00ffff;font-size:14px;'><b>Événements par pays :</b> {pays_count_str}</span>", unsafe_allow_html=True)

                st.write("")
                # ------ Table sans couleur ------
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=min(600, 50 + 26*min(len(df), 20))
                )
                # ------ Export CSV ------
                st.markdown("#### Exporter le calendrier filtré")
                st.download_button(
                    label="📥 Télécharger en CSV",
                    data=df.to_csv(index=False).encode(),
                    file_name="calendrier_economique.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Aucun événement trouvé pour ces critères.")
    else:
        st.warning("Veuillez sélectionner une plage de dates valide.")
with tab1:
    st.markdown("""
    <style>
    .bio-box {
        background: linear-gradient(120deg, #22293b 0%, #1e485a 100%);
        border: 2.5px solid #00ffd7bb;
        border-radius: 24px;
        padding: 36px 32px 26px 32px;
        margin: 35px auto 42px auto;
        max-width: 550px;
        box-shadow: 0 0 35px 5px #00ffd766, 0 0 80px 20px #00ffd722 inset;
        color: #f3faff;
        font-size: 1.15rem;
        line-height: 1.85;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
        text-align: center;
        position: relative;
        transition: box-shadow 0.2s;
    }
    .bio-box h2 {
        color: #00ffd7;
        font-size: 2.1rem;
        margin-bottom: 0.5em;
        letter-spacing: 1px;
        font-weight: 800;
    }
    .bio-box .subtitle {
        color: #ffea00;
        font-size: 1.18em;
        font-weight: 600;
        margin-bottom: 0.6em;
        letter-spacing: 0.2px;
    }
    .bio-box .bio-detail {
        margin: 1.1em 0 0.7em 0;
        font-size: 1.12em;
    }
    .bio-box .bio-contact {
        margin-top: 0.65em;
        font-size: 1.09em;
        color: #00ffd7;
    }
    .bio-box .bio-contact span {
        display: block;
        margin-bottom: 0.17em;
        color: #fff;
    }
    .bio-box .bio-decor {
        position: absolute;
        top: -18px; left: 50%;
        transform: translateX(-50%);
        background: #00ffd7;
        border-radius: 50%;
        width: 38px; height: 38px;
        box-shadow: 0 0 12px #00ffd7cc, 0 0 24px #00ffd766;
        display: flex;
        align-items: center; justify-content: center;
        font-size: 1.70rem;
        color: #001c18;
        border: 2.5px solid #fff;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bio-box">
        <div class="bio-decor">👩‍💼</div>
        <h2>Qui suis-je ?</h2>
        <div class="subtitle">TAOUDI EL IDRISSI HANANE</div>
        <div class="bio-detail">
            Étudiante en dernière année du cycle d'ingénieur en <b>Finance et Ingénierie Décisionnelle</b><br>
            à l'<b>École Nationale des Sciences Appliquées d'Agadir</b>.<br><br>
            Passionnée par la <b>finance de marché</b>, la <b>modélisation</b> et l'<b>innovation numérique</b>.
        </div>
        <div class="bio-contact">
            <span>📧 <b>Email</b> : hananetaoudielidrissi@gmail.com</span>
            <span>📞 <b>Téléphone</b> : +212 767857409</span>
        </div>
    </div>
    """, unsafe_allow_html=True)       
    st.markdown("""
    <style>
    .neon-box-fixed {
        background-color: #0f0f0f;
        border-left: 5px solid #00FFFF;
        padding: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 0 10px #00FFFF;
        color: white;
        font-weight: bold;
    }
    .footer-fixed {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.9rem;
        color: #aaaaaa;
    }
    </style>
    
    ### 🚀 Que pouvez-vous faire ici ?

    <span style='color: white; font-size: 1.05rem;'>

    <div class="neon-box-fixed">
    • <b>Consulter le Calendrier Économique</b>  
    Suivez les annonces macro-finance majeures (taux directeurs, PIB, PMI…) pour planifier vos couvertures FX.
    </div>

    <div class="neon-box-fixed">
    • <b>Visualiser MarketPulse : FX-DashBoard</b>  
    Obtenez les taux EUR/MAD, USD/MAD et EUR/USD en temps réel et leurs historiques, accompagnés d’indicateurs clés (ouverture, clôture, spreads, variation 1 an).
    </div>
    <div class="neon-box-fixed">
    • <b>Price Forward FX Instantanément</b>  
    Déterminez le taux forward en quelques clics selon la couverture de taux d’intérêt, avec affichage des pips et indication “REPORT”/“DÉPORT”.
    </div>
    <div class="neon-box-fixed">
    • <b>Valoriser vos Options de Change via différents modèles</b>  
    Accédez aux modèles Black-Scholes, Binomial, Trinomial, Monte Carlo, visualisez arbres et greeks en temps réel.
    </div>

     <div class="neon-box-fixed">
    • <b>Analyser vos Stratégies FX Options</b>  
    Testez instantanément diverses stratégies optionnelles (Short/Long Call-Put, Spreads, Straddles, Strangles) sur le marché des changes : visualisez le payoff, le P&L dynamique et les chiffres clés pour chaque scénario.
     </div>           
    <div class="neon-box-fixed">
    • <b>Calculer la Volatilité Implicite et le Smile</b>  
    Mesurez la volatilité du marché via le modèle Garman-Kohlhagen, générez et visualisez votre volatilité implicite et votre smile de volatilité.
    </div>

    <div class="neon-box-fixed">
    • <b>Simuler un Tunnel de Change (symétrique ou asymétrique)</b>
    Visualisez et comparez le pricing, les scénarios à maturité et le payoff de votre stratégie tunnel (plancher/plafond, ratios, maturité…).
    </div>

    <div class="neon-box-fixed">
    • <b>Analyser une Option Participative de Change</b>
    Calculez la prime, comparez-la à une option vanille, visualisez le payoff, et mesurez l’impact de la volatilité : tout pour évaluer l’intérêt d’une protection partielle sur le marché des devises.
    </div>                        

    <div class="neon-box-fixed">
    • <b>Valoriser un Currency Swap</b>  
    Chargez vos courbes de taux, paramétrez votre swap (EUR/MAD ou USD/MAD), et obtenez le prix net ainsi que le DV01 en MAD.
    </div>

    <div class="neon-box-fixed">
    • <b>Simuler des Scénarios de Stress FX</b>  
    Testez la résistance de votre portefeuille multi-devises face à la dévaluation du MAD et à l’appréciation de l’EUR/USD. Visualisez waterfall, sensibilité 2D, grille de cas et distribution de pertes.
    </div>

    </span>

    <div class="footer-fixed">
    Développé avec ❤️ par Hanane - Stage AWB SDM
    </div>
                
    <hr>
    <div style='text-align: center; font-size: 0.9rem;'>
        © 2025 TAOUDI EL IDRISSI Hanane. Ce logiciel est protégé par le droit d’auteur.<br>
    </div>
    <hr>            
    """, unsafe_allow_html=True)

with tab4:
    st.title("Contrat à Terme - FX FORWARD")
    import streamlit as st
    import numpy as np
    import plotly.graph_objects as go
    import pandas as pd

    # ------ Texte d'utilité affiché en permanence ------
    st.markdown("""
    <style>
    .utilite-box {
        background: rgba(10, 18, 35, 0.85);
        border: 2px solid #00ffff77;
        border-radius: 18px;
        padding: 17px 25px;
        margin-bottom: 22px;
        margin-top: 10px;
        box-shadow: 0 0 18px 2px #00ffff33, 0 0 35px 10px #00ffff22 inset;
        color: white;
        font-size: 16px;
        line-height: 1.55;
    }
    </style>
    <div class="utilite-box">
    <b>Ce module permet de calculer et d’analyser le taux de change à terme (forward FX) entre deux devises.</b><br>
    <ul>
        <li>Calculez le taux forward à partir des paramètres de marché (spot, taux, durée...)</li>
        <li>Obtenez automatiquement les points de swap et la situation de marché (report/déport)</li>
        <li>Analysez la sensibilité du taux à la variation des taux d’intérêt et à l’horizon</li>
        <li>Visualisez la courbe forward et exportez vos scénarios au besoin</li>
        <li>Comprenez le rôle du contrat forward en gestion du risque de change</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # ------ Checkbox pour afficher la description complète ------
    show_desc = st.checkbox("Afficher les détails sur les contrats forwards de change")

    if show_desc:
        st.markdown("""
        <style>
        .intro-box {
            background: rgba(20, 20, 30, 0.7);
            border: 2px solid rgba(0, 255, 255, 0.4);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 0 20px rgba(0,255,255,0.3), 0 0 40px rgba(0,255,255,0.2);
            backdrop-filter: blur(10px);
            color: white;
            font-size: 17px;
            line-height: 1.6;
        }
        </style>
        """, unsafe_allow_html=True)

        forward_html = """
        <div class="intro-box">
        <h2>📖 Définition du Contrat Forward de Change</h2>
        <p>Le change à terme est un accord portant sur l’achat ou la vente d’un montant déterminé de devises, à un cours fixé immédiatement, mais le règlement et la livraison ont lieu à une date d’échéance future précise.</p>
        <p>Le change à terme est un engagement ferme et définitif qui porte sur la quantité de devises, le cours de change et la date d’échéance. Ces éléments sont négociés le jour de l’accord, alors que les mouvements de trésorerie sont effectués le jour de l’échéance.</p>
        <p>L’intérêt de cet accord découle de la fixation, dès sa conclusion, du prix auquel sera exécutée l’opération de change à terme. Ainsi, le risque de change lié à une évolution défavorable des cours est éliminé.</p>
        <p>Le change à terme est une technique de couverture qui permet de figer le cours de change, mais de renoncer à un gain éventuel en cas d’évolution favorable des cours.</p>

        <h2>🔍 Explication des Situations de Marché</h2>
        <h4>Le Phénomène de Report (Contango)</h4>
        <p>Le report se manifeste lorsque le taux forward dépasse le taux spot. Cette situation typique se produit lorsque la devise de base présente des taux d'intérêt inférieurs à ceux de la devise cotée.</p>
        <p><em>Exemple :</em> Dans le cas du yen japonais (taux à 0.1%) contre le dollar australien (taux à 4.5%) pour une échéance d'un an, le forward affiche systématiquement un premium pouvant atteindre 4%.</p>
        <p>Les acteurs du marché interprètent ce report comme le coût de portage nécessaire pour maintenir la position.</p>

        <h4>Le Cas de Déport (Backwardation)</h4>
        <p>À l'inverse, le déport apparaît sur les devises à haut rendement ou soumises à des risques politiques.</p>
        <p><em>Exemple :</em> Le réal brésilien en est une illustration parlante : malgré des taux directeurs élevés (13% en 2023), les forwards BRL/USD présentent souvent un discount.</p>
        <p>Ce paradoxe s'explique par la prime de risque exigée par les investisseurs pour détenir des actifs brésiliens, qui dépasse l'avantage du différentiel de taux.</p>

        <h2>🏗 Structure et Fonctionnement</h2>
        <p>Voir tableau ci-dessous ⬇️</p>

        <h2>🎯 Enjeux Stratégiques</h2>
        <b>Avantages</b>
        <ul>
            <li>Protection contre le risque de change en éliminant l'incertitude liée aux fluctuations du marché.</li>
            <li>Sécurisation des coûts et revenus grâce à la fixation anticipée du taux.</li>
            <li>Adaptabilité de l'accord selon les besoins spécifiques de l'entreprise ou de l'investisseur.</li>
        </ul>
        <b>Inconvénients & Risques</b>
        <ul>
            <li>Renonciation à un gain éventuel en cas d'évolution favorable des cours.</li>
            <li>Engagement ferme et irrévocable jusqu'à l'échéance, sauf en cas de résiliation anticipée avec frais.</li>
            <li>Risque de contrepartie si l'une des parties ne respecte pas ses engagements.</li>
        </ul>

        <h2>🔢 Formalisation Mathématique du Taux Forward</h2>
        <p>Formule :</p>
        </div>
        """
        st.markdown(forward_html, unsafe_allow_html=True)
        st.latex(r"F(T) = S_0 \times \frac{1 + r_{quote} \times T}{1 + r_{base} \times T}")

        st.markdown("""
        <div class="intro-box">
        <h3>📊 Explication des variables</h3>
        <ul>
        <li><b>F(T)</b> : Taux de change forward</li>
        <li><b>S₀</b> : Taux de change spot (aujourd’hui)</li>
        <li><b>r<sub>quote</sub></b> : Taux d’intérêt de la devise de cotation</li>
        <li><b>r<sub>base</sub></b> : Taux d’intérêt de la devise de base</li>
        <li><b>T</b> : Durée du contrat</li>
        </ul>
        <p>Cette formule repose sur la <strong>parité des taux d’intérêt couverte</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="intro-box"><h3>🧱 Étapes d’un Contrat Forward</h3></div>', unsafe_allow_html=True)
        df_etapes = pd.DataFrame({
            "Étape": ["Signature", "Fixation du Taux", "Échéance"],
            "Description": [
                "Définition des paramètres : paires de devises, montant, date d’échéance et calcul du taux forward.",
                "Détermination du taux forward en fonction du taux spot et des taux d’intérêt des devises concernées.",
                "À la date convenue, l’échange de devises a lieu selon les termes fixés à la signature."
            ]
        })
        st.table(df_etapes)

        st.markdown("""
        <div class="intro-box">
        <h2>Récapitulation :</h2>
        <ul>
        <li>Le contrat forward de change est un outil essentiel pour la gestion du risque de change.</li>
        <li>Il permet de verrouiller un taux de change afin d'éviter l'impact des fluctuations du marché, tout en impliquant certains engagements irrévocables et risques.</li>
        <li>La compréhension des mécanismes tels que le report (contango) et le déport (backwardation) est cruciale pour optimiser son utilisation.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # ------ Paramètres de Marché ------
    st.subheader("🔧 Paramètres de Marché")
    param_cols = st.columns([2, 1, 2, 1, 2])

    with param_cols[0]:
        spot_rate = st.number_input(
            "Cours Spot (S)",
            value=1.1200,
            format="%.4f",
            help="Taux de change spot actuel (devise de base contre devise de cotation)."
        )

    with param_cols[1]:
        days = st.number_input(
            "Jours (N)",
            value=90,
            help="Nombre de jours jusqu'à l'échéance du contrat."
        )

    with param_cols[2]:
        base_ccy_rate = st.number_input(
            "Taux Devise Base (%)",
            value=2.5,
            help="Taux d'intérêt de la devise de base (ex : EUR pour EUR/USD)."
        ) / 100

    with param_cols[3]:
        term_ccy_rate = st.number_input(
            "Taux Devise Cotation (%)",
            value=1.75,
            help="Taux d'intérêt de la devise de cotation (ex : USD pour EUR/USD)."
        ) / 100

    with param_cols[4]:
        convention = st.selectbox(
            "Convention de Jours",
            ["Act/360", "Act/365"],
            help="Choix de la convention de calcul des jours pour le taux d'intérêt."
        )

    # ------ Robustesse inputs ------
    if spot_rate <= 0 or days <= 0:
        st.error("Le spot et le nombre de jours doivent être strictement positifs.")
        st.stop()
    if not (-0.1 < base_ccy_rate < 0.2) or not (-0.1 < term_ccy_rate < 0.2):
        st.error("Les taux doivent être compris entre -10% et +20%.")
        st.stop()

    # ------ Calculs ------
    def calculate_day_fraction(days, convention):
        if convention == "Act/360":
            return days / 360
        if convention == "Act/365":
            return days / 365
        return days / 360

    day_fraction = calculate_day_fraction(days, convention)
    forward_rate = spot_rate * (1 + term_ccy_rate * day_fraction) / (1 + base_ccy_rate * day_fraction)
    forward_points = (forward_rate - spot_rate) * 10000

    st.subheader("📈 Résultats du Pricing")
    # ------ Résultats stylés ------
    result_cols = st.columns(3)

    with result_cols[0]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='header'>Forward Rate</div>
            <div style='font-size: 2rem;'>{forward_rate:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with result_cols[1]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='header'>Swap Points</div>
            <div style='font-size: 2rem;'>{forward_points:.1f} pips</div>
        </div>""", unsafe_allow_html=True)

    with result_cols[2]:
        status = "REPORT" if forward_rate > spot_rate else "DÉPORT"
        color = "#00FF00DA" if forward_rate > spot_rate else "#FF4500"
        st.markdown(f"""
        <div class='metric-box'>
            <div class='header'>Situation</div>
            <div style='font-size: 2rem; color: {color}'>{status}</div>
        </div>""", unsafe_allow_html=True)

    # ------ Explication pédagogique des résultats ------
    st.markdown(f"""
    <div class="utilite-box" style="margin-top:10px">
    <b>Interprétation :</b><br>
    - <b>Forward Rate</b> : Taux obtenu pour l’échange à terme, calculé selon la parité des taux d’intérêt.<br>
    - <b>Swap Points</b> : Différence (en pips) entre le taux forward et le spot. <br>
    - <b>{status}</b> : {'Le forward est supérieur au spot (prime à payer).' if status=='REPORT' else 'Le forward est inférieur au spot (décote).' }
    </div>
    """, unsafe_allow_html=True)

    # ------ Visualisations ------
    st.subheader("📊 Analyse de Sensibilité")

    sens_cols = st.columns(2)

    with sens_cols[0]:
        st.markdown("<span class='info-bulle'>Faites varier <b>le taux d'intérêt de la devise de cotation</b> (autre taux fixe).</span>", unsafe_allow_html=True)
        rate_range = st.slider("Variation du taux de cotation (bps)", -200, 200, 100)
        rates = np.linspace(term_ccy_rate - rate_range/10000, term_ccy_rate + rate_range/10000, 50)
        scenario_forwards = spot_rate * (1 + rates * day_fraction) / (1 + base_ccy_rate * day_fraction)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=(rates*100), y=scenario_forwards, name='Sensibilité'))
        fig.update_layout(
            title="Impact de la variation du taux de cotation",
            xaxis_title="Taux de cotation (%)",
            yaxis_title="Taux Forward"
        )
        st.plotly_chart(fig, use_container_width=True)

    with sens_cols[1]:
        st.markdown("<span class='info-bulle'>Faites varier <b>l'horizon (jours)</b> pour visualiser la courbe forward.</span>", unsafe_allow_html=True)
        time_range = st.slider("Horizon (jours)", 30, 730, 365)
        days_range = np.arange(1, time_range + 1)
        forwards = [spot_rate * (1 + term_ccy_rate * calculate_day_fraction(d, convention)) /
                    (1 + base_ccy_rate * calculate_day_fraction(d, convention)) for d in days_range]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=days_range, y=forwards, name='Courbe Forward'))
        fig2.add_hline(y=spot_rate, line_dash="dash", annotation_text="Spot")
        fig2.update_layout(
            title="Évolution du taux forward dans le temps",
            xaxis_title="Jours",
            yaxis_title="Taux Forward"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ------ Export CSV des scénarios (optionnel) ------
    if st.checkbox("Exporter les scénarios (CSV)"):
        df_export = pd.DataFrame({
            "jours": days_range,
            "forward_rate": forwards
        })
        st.download_button(
            label="Télécharger la courbe forward (CSV)",
            data=df_export.to_csv(index=False).encode(),
            file_name="courbe_forward.csv",
            mime="text/csv"
        )    




# 📦 Imports
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


with tab9:
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    st.title("FX BOOK - Options Participatives ")
    # ------ Explication du module en haut ------
    st.markdown("""
    <div style="background: rgba(15, 23, 45, 0.95); border: 2.5px solid #00ffd7cc; border-radius: 18px;
         padding: 24px 26px; margin-bottom: 24px; color: #e8fcff; font-size:18px; font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;">
        <b>Ce module vous permet de simuler et d'analyser le pricing d'une option participative de change :</b>
        <ul>
            <li>Calculez la prime d'une option participative selon vos paramètres de marché</li>
            <li>Comparez-la à une option vanille classique</li>
            <li>Visualisez le payoff et l'impact de la volatilité</li>
            <li>Exportez vos résultats</li>
        </ul>
        <i>Idéal pour comprendre l'intérêt des produits dérivés "participatifs" en gestion du risque de change.</i>
    </div>
    """, unsafe_allow_html=True)

    # ------ Bouton pour afficher ou masquer la description pédagogique ------
    show_desc = st.button("Afficher / masquer la description des options participatives")
    if show_desc:
        st.markdown("""
        <div class="intro-box" style='margin-bottom:20px;'>
            <h2>📘 Option Participative de Change : Définition & Fonctionnement</h2>
            <p>
                L'<strong>option participative de change</strong> est une option vanille (call ou put) à laquelle est associée un <strong>taux de participation</strong> favorable à l’acheteur.<br>
                Elle permet de se protéger tout en bénéficiant partiellement d’une performance favorable du marché.
            </p>
            <table class="neon-table">
                <tr><th>Type</th><td>Vanille (Call/Put) + clause de participation favorable</td></tr>
                <tr><th>Devises</th><td>Ex : MAD/EUR ou MAD/USD</td></tr>
                <tr><th>Participation</th><td>L’acheteur bénéficie d’un pourcentage des gains au-delà du strike</td></tr>
                <tr><th>Prime</th><td>Réduite par rapport à l’option standard</td></tr>
                <tr><th>Scénarios</th>
                    <td>
                        <b>Call participatif :</b><br>
                        - Si Spot &lt; Strike : Pas d’exercice, pas de gain<br>
                        - Si Spot &gt; Strike : Gain = (Spot - Strike) × Participation × Nominal<br>
                        <b>Put participatif :</b><br>
                        - Si Spot &gt; Strike : Pas d’exercice, pas de gain<br>
                        - Si Spot &lt; Strike : Gain = (Strike - Spot) × Participation × Nominal
                    </td>
                </tr>
            </table>
            <ul>
                <li>Protection contre l’évolution défavorable du taux de change</li>
                <li>Participation partielle à la performance favorable</li>
                <li>Prime moins élevée (ou nulle) qu’une option classique</li>
            </ul>
            <strong>Exemple :</strong><br>
            <span style="color:#e6ffb9;">• Participation 50%, Strike 11, Nominal 100 000</span><br>
            <span style="color:#e6ffb9;">• Si Spot = 12 à l’échéance : Gain = (12 - 11) × 0.5 × 100 000 = 50 000 MAD</span>
        </div>
        """, unsafe_allow_html=True)

    # ------ FONCTIONS PRICING ------
    def garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type):
        if T <= 0:
            raise ValueError("La maturité doit être positive")
        if sigma <= 0:
            raise ValueError("La volatilité doit être positive")
        if S <= 0 or K <= 0 or np.isnan(S) or np.isnan(K):
            raise ValueError("Spot et Strike doivent être positifs")
        d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if option_type == 'call':
            return S * np.exp(-r_f*T) * norm.cdf(d1) - K * np.exp(-r_d*T) * norm.cdf(d2)
        elif option_type == 'put':
            return K * np.exp(-r_d*T) * norm.cdf(-d2) - S * np.exp(-r_f*T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")

    def participative_option_price(S, K, T, r_d, r_f, sigma, nominal, participation, option_type):
        vanilla_premium = garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type)
        return vanilla_premium * participation * nominal

    def plot_payoff(S, K, nominal, participation, option_type, spot_min=None, spot_max=None):
        spot_min = spot_min or (0.7 * K)
        spot_max = spot_max or (1.3 * K)
        S_vals = np.linspace(spot_min, spot_max, 150)
        if option_type == 'call':
            payoff = np.where(S_vals > K, (S_vals - K) * participation * nominal, 0)
        else:
            payoff = np.where(S_vals < K, (K - S_vals) * participation * nominal, 0)
        fig, ax = plt.subplots()
        ax.plot(S_vals, payoff, label="Payoff participatif", color="#099")
        ax.axvline(K, linestyle="--", color="grey", label="Strike")
        ax.set_xlabel("Spot à l'échéance")
        ax.set_ylabel("Gain (€ ou MAD)")
        ax.set_title("Payoff à l'échéance de l'option participative")
        ax.legend()
        ax.grid(True, alpha=0.2)
        return fig

    def graph_vol_sensitivity_part(S, K, T, r_d, r_f, vol_range, nominal, participation, option_type):
        premiums = []
        for sigma in vol_range:
            vanilla = garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type)
            premiums.append(vanilla * participation * nominal)
        fig, ax = plt.subplots()
        ax.plot(vol_range, premiums, color="#c28cfa")
        ax.set_xlabel("Volatilité")
        ax.set_ylabel("Prime participative")
        ax.set_title("Prime vs. Volatilité")
        ax.grid(True, alpha=0.2)
        return fig

    # ------ SAISIE PARAMÈTRES UTILISATEUR (3 par ligne, nominal sous spot, volatilité sous participation) ------
    colA, colB, colC = st.columns(3)
    with colA:
        option_type = st.selectbox("Type d'option", ["call", "put"])
        S = st.number_input("Taux Spot (S)", value=11.0, step=0.01, min_value=0.0001)
        nominal = st.number_input("Nominal (montant à couvrir)", value=100_000.0, min_value=1.0)
    with colB:
        K = st.number_input("Strike (prix d'exercice)", value=11.0, step=0.01, min_value=0.0001)
        participation = st.slider("Taux de participation (%)", 1, 100, 50) / 100
        sigma = st.number_input("Volatilité (ex: 0.12)", value=0.12, min_value=0.0001)
    with colC:
        T = st.number_input("Maturité (en années)", value=0.5, min_value=0.01)
        r_d = st.number_input("Taux domestique (MAD)", value=0.03)
        r_f = st.number_input("Taux étranger (EUR/USD)", value=0.01)

    st.markdown("""
    <details>
    <summary><b>Voir la formule du pricing (modèle Garman-Kohlhagen)</b></summary>
    <div style="background:#222b37;border-radius:12px;padding:13px 20px 10px 20px;margin:8px 0 15px 0;">
    <b>Prime (vanille) : </b>
    <br>
    <span style="color:#00ffd7;">
    Call = S × exp(-r<sub>f</sub>T) × N(d₁) – K × exp(-r<sub>d</sub>T) × N(d₂)
    <br>
    Put = K × exp(-r<sub>d</sub>T) × N(–d₂) – S × exp(-r<sub>f</sub>T) × N(–d₁)
    </span>
    <br>
    <b>Avec :</b><br>
    d₁ = [ln(S/K) + (r<sub>d</sub> – r<sub>f</sub> + ½σ²)T] / (σ√T) <br>
    d₂ = d₁ – σ√T
    <br><br>
    <b>Prime participative = Prime vanille × taux de participation × nominal</b>
    </div>
    </details>
    """, unsafe_allow_html=True)

    # ------ CALCUL & RÉSULTATS ------
    if st.button("💰 Afficher les Résultats Participatifs", key="btn_tab9_participative"):
        st.header("Résultats du Pricing Participatif")
        try:
            prime = participative_option_price(S, K, T, r_d, r_f, sigma, nominal, participation, option_type)
            vanilla = garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type) * nominal
            st.markdown("### 📋 Résultats")
            st.markdown(
                f"""
                <table class="neon-table">
                <tr><th>Prime participative</th><td>{round(prime,2):,.2f} MAD</td></tr>
                <tr><th>Prime classique (vanille)</th><td>{round(vanilla,2):,.2f} MAD</td></tr>
                <tr><th>Participation</th><td>{int(participation*100)} %</td></tr>
                <tr><th>Type</th><td>{"Call" if option_type=="call" else "Put"}</td></tr>
                </table>
                """, unsafe_allow_html=True
            )
            # Export possible
            st.download_button(
                label="Exporter les résultats (CSV)",
                data=f"Prime participative,{prime}\nPrime classique,{vanilla}\nParticipation,{participation}\nType,{option_type}",
                file_name="resultat_participative.csv",
                mime="text/csv"
            )

            # Graphe payoff à l'échéance
            st.subheader("📈 Payoff à l'échéance")
            fig_payoff = plot_payoff(S, K, nominal, participation, option_type)
            st.pyplot(fig_payoff, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")
with tab6:
    import streamlit as st
    import numpy as np
    import plotly.graph_objs as go
    from scipy.stats import norm

   
    st.title("FX Volatilité – Volatilité Implicite & Smile")
     # ------ Encadrement NEON pour la description ------
    st.markdown("""
    <style>
    .neon-desc {
        background: rgba(15, 15, 30, 0.92);
        border: 2px solid rgba(0, 255, 255, 0.7);
        border-radius: 20px;
        padding: 22px 30px;
        margin-bottom: 30px;
        margin-top: 10px;
        box-shadow: 0 0 18px 2px #00ffff55, 0 0 45px 10px #00ffff22 inset;
        color: white;
        font-size: 17px;
        line-height: 1.65;
        backdrop-filter: blur(5px);
    }
    </style>
    <div class="neon-desc">
    Ce module permet de <b>calculer et analyser la volatilité implicite d'une option de change</b> (FX) à partir d'un prix de marché.<br>
    <ul>
        <li>Calculez la volatilité implicite à partir du prix d'une option Call ou Put</li>
        <li>Visualisez le <b>smile de volatilité</b> réel ou simulé sur une gamme de strikes</li>
        <li>Observez la <b>convergence réelle</b> de la méthode de Newton-Raphson</li>
        <li>Analysez la <b>sensibilité du prix à la volatilité</b> (Vega)</li>
        
    </ul>
    </div>
    """, unsafe_allow_html=True)


    # 🎯 Fonctions utiles

    # Garman-Kohlhagen Call
    def fx_call_price_BS_model(spot, vol, rd, rf, T, K):
        d1 = (np.log(spot/K) + (rd - rf + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        price = np.exp(-rd*T) * (spot*np.exp((rd - rf)*T)*norm.cdf(d1) - K*norm.cdf(d2))
        return price

    # Garman-Kohlhagen Put
    def fx_put_price_BS_model(spot, vol, rd, rf, T, K):
        d1 = (np.log(spot/K) + (rd - rf + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        price = np.exp(-rd*T) * (K*norm.cdf(-d2) - spot*np.exp((rd - rf)*T)*norm.cdf(-d1))
        return price

    # Calcul de volatilité implicite via Newton-Raphson, avec stockage de la convergence
    def fx_implied_volatility(option_type, spot, strike, rd, rf, T, market_price, initial_vol=0.2, maxIter=100, epsilon=1e-6):
        dVol = 1e-6
        vol = initial_vol
        errors = []
        vols = []
        for i in range(maxIter):
            if option_type == "Call":
                price = fx_call_price_BS_model(spot, vol, rd, rf, T, strike)
                price_minus = fx_call_price_BS_model(spot, vol - dVol, rd, rf, T, strike)
            else:
                price = fx_put_price_BS_model(spot, vol, rd, rf, T, strike)
                price_minus = fx_put_price_BS_model(spot, vol - dVol, rd, rf, T, strike)
            diff = price - market_price
            errors.append(abs(diff))
            vols.append(vol)
            derivative = (price - price_minus) / dVol
            if abs(derivative) < epsilon or abs(diff) < epsilon:
                break
            vol -= diff / derivative
            # Garde la volatilité dans la plage raisonnable
            if vol < 0.0001 or vol > 5:
                break
        return vol, errors, vols

    # Vega - Sensibilité du prix à la vol
    def fx_option_vega(spot, vol, rd, rf, T, K):
        d1 = (np.log(spot/K) + (rd - rf + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        vega = spot * np.exp((rd - rf)*T) * norm.pdf(d1) * np.sqrt(T) * np.exp(-rd*T)
        return vega

    # 🎯 Inputs utilisateurs
    col1, col2, col3 = st.columns(3)

    with col1:
        spot = st.number_input("Spot (MAD par devise)", value=10.00)
        strike = st.number_input("Strike (K)", value=10.20)
        option_type = st.selectbox("Option Type", ["Call", "Put"],key="vol_tab6")

    with col2:
        rd = st.number_input("Taux Domestique (MAD)", value=0.03)
        rf = st.number_input("Taux Etranger (USD ou EUR)", value=0.05)
        time_to_maturity = st.number_input("Maturité (années)", value=0.25)

    with col3:
        market_price = st.number_input("Prix du Marché de l'Option", value=0.15)
        initial_vol = st.number_input("Volatilité Initiale Estimée", value=0.2)

    st.markdown("---")

    # 📈 Calcul principal (avec stockage des erreurs pour la convergence)
    try:
        implied_vol, errors, vols_NR = fx_implied_volatility(
            option_type, spot, strike, rd, rf, time_to_maturity, market_price, initial_vol)
        if implied_vol < 0.0001 or implied_vol > 5:
            st.error("❌ Newton-Raphson n'a pas convergé vers une volatilité raisonnable. Vérifiez vos paramètres ou le prix de marché.")
        else:
            st.success(f"✅ Volatilité Implicite Calculée : **{implied_vol*100:.2f}%**")
    except Exception as e:
        st.error("Erreur lors du calcul de la volatilité implicite : " + str(e))
        implied_vol = np.nan
        errors = []
        vols_NR = []

    # 🎨 Smile de volatilité réel ou simulé (avec option)
    st.subheader("🎯 Smile de Volatilité (Strike vs Volatilité)")
    simulate_smile = st.checkbox("Simuler des prix de marché avec smile (volatilité variable)", value=True)

    strike_range = np.linspace(spot * 0.8, spot * 1.2, 35)
    smile_vols = []

    if simulate_smile:
        beta = 4  # intensité du sourire
        for K in strike_range:
            # Volatilité variable selon le strike (parabole)
            vol_simul = implied_vol * (1 + beta * ((K/spot - 1)**2))
            # Prix de marché simulé avec cette volatilité variable
            if option_type == "Call":
                price_simul = fx_call_price_BS_model(spot, vol_simul, rd, rf, time_to_maturity, K)
            else:
                price_simul = fx_put_price_BS_model(spot, vol_simul, rd, rf, time_to_maturity, K)
            # Retrouve la vol implicite à partir de ce prix simulé
            IV, _, _ = fx_implied_volatility(option_type, spot, K, rd, rf, time_to_maturity, price_simul, implied_vol)
            smile_vols.append(IV * 100)
    else:
        for K in strike_range:
            if option_type == "Call":
                price_at_central_vol = fx_call_price_BS_model(spot, implied_vol, rd, rf, time_to_maturity, K)
            else:
                price_at_central_vol = fx_put_price_BS_model(spot, implied_vol, rd, rf, time_to_maturity, K)
            IV, _, _ = fx_implied_volatility(option_type, spot, K, rd, rf, time_to_maturity, price_at_central_vol, implied_vol)
            smile_vols.append(IV * 100)

    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(
        x=strike_range, y=smile_vols, mode='lines+markers',
        line=dict(color="cyan", width=3),
        marker=dict(size=6)
    ))
    fig_smile.update_layout(
        template="plotly_dark",
        xaxis_title="Strike",
        yaxis_title="Volatilité Implicite (%)",
        title="Smile de volatilité (réel ou simulé)",
        margin={"l":50,"r":30,"t":60,"b":40}
    )
    st.plotly_chart(fig_smile, use_container_width=True)

    # ➡️ Courbe de convergence Newton-Raphson réelle
    st.subheader("🎯 Convergence Newton-Raphson (Erreur à chaque itération)")
    if errors:
        fig_convergence = go.Figure()
        fig_convergence.add_trace(go.Scatter(
            x=list(range(1, len(errors)+1)),
            y=errors,
            mode='lines+markers',
            line=dict(color="lime", width=3),
            marker=dict(size=8)
        ))
        fig_convergence.update_layout(
            template="plotly_dark",
            xaxis_title="Itérations",
            yaxis_title="Erreur absolue (prix)",
            title="Convergence Newton-Raphson"
        )
        st.plotly_chart(fig_convergence, use_container_width=True)
    else:
        st.info("Aucune convergence Newton-Raphson à afficher.")

    # ➡️ Sensibilité prix de l’option à la volatilité (Vega)
    st.subheader("🎯 Sensibilité : Prix de l'Option en fonction de la volatilité (Vega)")
    vol_vec = np.linspace(0.01, max(implied_vol*2, 0.5), 40)
    price_vec = []
    vega_vec = []
    for v in vol_vec:
        if option_type == "Call":
            price = fx_call_price_BS_model(spot, v, rd, rf, time_to_maturity, strike)
        else:
            price = fx_put_price_BS_model(spot, v, rd, rf, time_to_maturity, strike)
        price_vec.append(price)
        vega_vec.append(fx_option_vega(spot, v, rd, rf, time_to_maturity, strike))

    fig_sensi = go.Figure()
    fig_sensi.add_trace(go.Scatter(
        x=vol_vec*100, y=price_vec, mode='lines', name="Prix", line=dict(color="orange", width=3)
    ))
    fig_sensi.add_trace(go.Scatter(
        x=vol_vec*100, y=vega_vec, mode='lines', name="Vega", line=dict(color="magenta", width=2, dash="dash")
    ))
    fig_sensi.update_layout(
        template="plotly_dark",
        xaxis_title="Volatilité (%)",
        yaxis_title="Prix / Vega",
        title="Sensibilité du prix de l'option à la volatilité (Vega)",
        legend=dict(x=0.65, y=0.98, bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_sensi, use_container_width=True)

    # Explications pédagogiques
    with st.expander("ℹ️ Explications "):
        st.markdown("""
        - <b>Volatilité implicite</b> : c'est la volatilité “déduite” du prix de l'option, en inversant la formule de Black-Scholes/Garman-Kohlhagen.<br>
        - <b>Smile de volatilité</b> : il représente la variation de la volatilité implicite selon le strike, illustrant la non-linéarité du marché. Avec la case cochée, on observe un vrai "sourire" grâce à la simulation.<br>
        - <b>Convergence Newton-Raphson</b> : la méthode numérique utilisée pour retrouver la volatilité implicite. Le graphe montre la vitesse et la précision de la convergence.<br>
        - <b>Sensibilité (Vega)</b> : le vega mesure la variation du prix de l'option pour une variation de volatilité. Plus le vega est élevé, plus le prix est sensible à la vol.<br>
        - <b>Conseil</b> : Si la convergence échoue, vérifiez vos paramètres ou le prix de marché : il se peut qu'il soit incompatible avec la configuration choisie.
        """, unsafe_allow_html=True)

with tab5:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import streamlit as st

    # ========== INPUTS (inchangés) ==========
    st.title("Pricing Option de Change (FX) - Garman-Kohlhagen et Arbres")
    st.markdown("""
    <div class="neon-box-fixed">
        <b>Ce module vous permet de calculer et comparer la valeur d'une option de change selon différents modèles :</b>
        <ul style="margin-left: -1.5em;">
            <li>Obtenez le prix théorique d'une option FX (call/put) avec Garman-Kohlhagen, arbres binomiaux, trinômiaux ou simulation Monte Carlo</li>
            <li>Adaptez facilement vos paramètres de marché (cours spot, strike, taux, volatilité, maturité, style...)</li>
            <li>Analysez la convergence des méthodes numériques et la sensibilité du prix de l’option aux paramètres clés</li>
            <li>Comparez les résultats entre modèles et visualisez leur comportement</li>
        </ul>
        <i>Un outil pédagogique et pratique pour la gestion ou la formation sur le pricing des options de change.</i>
    </div>
    """, unsafe_allow_html=True)

    col0, col1, col2 = st.columns([1, 2, 2])
    with col0:
        st.markdown("### Devises")
        devise_domestique = st.selectbox(
            "Devise domestique (taux r_d)",
            options=["MAD"],
            index=0,
            key="devise_domestique_tab5")
        devise_etrangere = st.selectbox(
            "Devise étrangère (taux r_f)",
            options=["EUR", "USD"],
            index=0,
            key="devise_etrangere_tab5")
        st.caption("Convention : r_d = taux sans risque pour la devise domestique (MAD), r_f = taux pour la devise étrangère (EUR/USD)")

    with col1:
        S = st.number_input(
            f"Spot FX (S) [{devise_domestique}/{devise_etrangere}]",
            min_value=0.0001,
            value=11.00,
            step=0.1,
            format="%.4f",
            key="spot_tab5")
        K = st.number_input(
            f"Strike (K) [{devise_domestique}/{devise_etrangere}]",
            min_value=0.0001,
            value=11.20,
            step=0.1,
            format="%.4f",
            key="strike_tab5")
        T = st.number_input(
            "Maturité (en années, ex: 0.5 pour 6 mois)",
            min_value=0.01,
            value=0.5,
            step=0.01,
            key="maturite_tab5")
        r_d = st.number_input(
            f"Taux sans risque domestique r_d ({devise_domestique})",
            value=0.035,
            step=0.001,
            format="%.4f",
            key="rd_tab5")

    with col2:
        r_f = st.number_input(
            f"Taux sans risque étranger r_f ({devise_etrangere})",
            value=0.025,
            step=0.001,
            format="%.4f",
            key="rf_tab5")
        sigma = st.number_input(
            "Volatilité annualisée (sigma)",
            min_value=0.0001,
            value=0.08,
            step=0.01,
            format="%.4f",
            key="vol_tab5")
        option_type = st.selectbox(
            "Type d'option",
            options=["call", "put"],
            key="option_type_tab5")
        option_style = st.selectbox(
            "Style d'option",
            options=["european", "american"],
            key="option_style_tab5")

    # ========== SLIDERS (2 à 2 côte à côte) ==========
    col_slider1, col_slider2 = st.columns(2)
    with col_slider1:
        steps = st.slider("Nombre de pas (binomial)", min_value=2, max_value=50, value=5)
        steps_tri = st.slider("Nombre de pas (trinomial)", min_value=2, max_value=30, value=5)
    with col_slider2:
        simulations = st.slider("Nombre de simulations MC", min_value=1000, max_value=200_000, value=100_000, step=1000)
        steps_paths = st.slider("Nombre d'étapes de simulation (pour tracé des chemins)", min_value=10, max_value=500, value=100, key="steps_paths_fx")
    # Second rangée sliders côte à côte
    col_slider3, col_slider4 = st.columns(2)
    with col_slider3:
        num_paths = st.slider("Nombre de chemins MC à visualiser", 100, 5000, 1000, key="num_paths_fx")
    with col_slider4:
        pass  # tu peux ajouter d'autres sliders ici si besoin

    # ========== MODELES (ajoute ici tes fonctions importées ou locales) ==========

    def garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type):
        if T <= 0:
            raise ValueError("La maturité doit être positive")
        d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if option_type == 'call':
            price = S * np.exp(-r_f*T) * norm.cdf(d1) - K * np.exp(-r_d*T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r_d*T) * norm.cdf(-d2) - S * np.exp(-r_f*T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")
        return price

    def greeks_garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type):
        if T <= 0:
            raise ValueError("La maturité doit être positive.")
        d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if option_type == 'call':
            delta = np.exp(-r_f * T) * norm.cdf(d1)
        elif option_type == 'put':
            delta = -np.exp(-r_f * T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'.")
        gamma = (np.exp(-r_f * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-r_f * T) * norm.pdf(d1) * np.sqrt(T)
        if option_type == 'call':
            theta = (
                - (S * np.exp(-r_f * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                - r_f * S * np.exp(-r_f * T) * norm.cdf(d1)
                + r_d * K * np.exp(-r_d * T) * norm.cdf(d2)
            ) / 365
        else:
            theta = (
                - (S * np.exp(-r_f * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                + r_f * S * np.exp(-r_f * T) * norm.cdf(-d1)
                - r_d * K * np.exp(-r_d * T) * norm.cdf(-d2)
            ) / 365
        if option_type == 'call':
            rho_d = T * K * np.exp(-r_d * T) * norm.cdf(d2) / 100
        else:
            rho_d = -T * K * np.exp(-r_d * T) * norm.cdf(-d2) / 100
        if option_type == 'call':
            rho_f = -T * S * np.exp(-r_f * T) * norm.cdf(d1) / 100
        else:
            rho_f = T * S * np.exp(-r_f * T) * norm.cdf(-d1) / 100
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho_d": rho_d,
            "rho_f": rho_f
        }

    # === Place ici tes fonctions binomial_tree_fx, trinomial_tree_fx, monte_carlo_fx, draw_tree, plot_convergence ===
    # --- Elles doivent être déjà définies/importées dans ton script principal ---

    # ========== CALCULS ==========
    try:
        prix_gk = garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type)
        prices_bi, values_bi = binomial_tree_fx(S, K, T, r_d, r_f, sigma, steps, option_type, option_style)
        prices_tri, values_tri = trinomial_tree_fx(S, K, T, r_d, r_f, sigma, steps_tri, option_type, option_style)
        prix_mc = monte_carlo_fx(S, K, T, r_d, r_f, sigma, option_type, int(simulations), 42)
        greeks = greeks_garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # ========== AFFICHAGE EN CARTES POUR LES PRIX ==========
    st.markdown("<h3 style='text-align: center; margin: 30px 0 10px 0; color: #2E3A59;'>📊 Résultats du Pricing (FX)</h3>", unsafe_allow_html=True)
    cols = st.columns(4)
    models = [
        ("Garman-Kohlhagen", prix_gk, "#4e79a7", f"{devise_domestique}"),
        ("Binomial FX", values_bi[0,0], "#e15759", f"{devise_domestique}"),
        ("Trinomial FX", values_tri[steps_tri,0], "#59a14f", f"{devise_domestique}"),
        ("Monte Carlo FX", prix_mc, "#f28e2b", f"{devise_domestique}")
    ]
    for (name, value, color, unit), col in zip(models, cols):
        with col:
            st.markdown(f"""
                <div style='
                    padding: 20px;
                    border-radius: 10px;
                    background: {color}10;
                    border-left: 4px solid {color};
                    margin: 10px 0;
                    text-align: center;
                '>
                    <h4 style='margin:0; color: {color};'>{name}</h4>
                    <h2 style='margin:5px 0; color: #2E3A59; font-weight:bold; font-size:2.4em;'>
                        {value:.4f} <span style='font-size:0.75em;'>{unit}</span>
                    </h2>
                </div>
            """, unsafe_allow_html=True)

    # ========== AFFICHAGE EN CARTES POUR LES GREEKS ==========
    st.markdown("<h3 style='text-align: center; margin: 40px 0 20px 0; color: #2E3A59;'>📉 Sensibilités (Greeks FX)</h3>", unsafe_allow_html=True)
    greek_list = [
        ("Δ Delta", greeks.get("delta",0), "#8cd17d"),
        ("Γ Gamma", greeks.get("gamma",0), "#86bcb6"),
        ("Θ Theta", greeks.get("theta",0), "#e15759"),
        ("ν Vega", greeks.get("vega",0), "#79706e"),
        (f"ρ<sub>d</sub> (dom)", greeks.get("rho_d",0), "#d7b5a6"),
        (f"ρ<sub>f</sub> (étr)", greeks.get("rho_f",0), "#ae87ff"),
    ]
    cols = st.columns(len(greek_list))
    for (name, value, color), col in zip(greek_list, cols):
        with col:
            st.markdown(f"""
                <div style='
                    padding: 15px;
                    border-radius: 8px;
                    background: {color}10;
                    text-align: center;
                    border: 1px solid {color}30;
                    margin: 5px 0;
                '>
                    <div style='font-size: 1.1em; color: {color}; font-weight: bold;'>{name}</div>
                    <div style='font-size: 1.35em; color: #2E3A59; font-weight:bold;'>{value:.6f}</div>
                </div>
            """, unsafe_allow_html=True)

    # ========== AFFICHAGE COTE A COTE DES ARBRES ==========
    st.markdown("<h3 style='text-align: center; margin: 40px 0 20px 0; color: #2E3A59;'>🌳 Visualisation des Arbres FX</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_bin = draw_tree(prices_bi, values_bi, steps, option_type, model='binomial')
        st.pyplot(fig_bin)
        st.caption("Arbre Binomial FX")
    with col2:
        fig_tri = draw_tree(prices_tri, values_tri, steps_tri, option_type, model='trinomial')
        st.pyplot(fig_tri)
        st.caption("Arbre Trinomial FX")

    # ========== AFFICHAGE COTE A COTE SIMULATIONS ET CONVERGENCE ==========
    st.markdown("<h3 style='text-align: center; margin: 40px 0 20px 0; color: #2E3A59;'>📈 Simulations & Convergence</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎲 Simulations Monte Carlo FX")
        dt = T / steps_paths
        fig_paths = plt.figure(figsize=(10, 6), facecolor='#f0f2f6')
        ax_paths = fig_paths.add_subplot(111)
        cmap = plt.get_cmap('Reds')
        for i in range(num_paths):
            path = np.zeros(steps_paths + 1)
            path[0] = S
            for j in range(1, steps_paths + 1):
                z = np.random.normal()
                path[j] = path[j-1] * np.exp((r_d - r_f - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            ax_paths.plot(path, color=cmap(i / num_paths), alpha=0.7, linewidth=0.9)
        ax_paths.set_title(f"Simulations de Monte Carlo - Chemins FX {devise_domestique}/{devise_etrangere}", fontsize=15, pad=15, color="#333333")
        ax_paths.set_xlabel("Étapes", fontsize=13, labelpad=10, color="#333333")
        ax_paths.set_ylabel(f"Taux de change spot ({devise_domestique}/{devise_etrangere})", fontsize=13, labelpad=10, color="#333333")
        ax_paths.grid(True, linestyle='--', alpha=0.5, color="#cccccc")
        ax_paths.set_facecolor('#f8f9fa')
        ax_paths.plot([], [], color='gray', label=f'{num_paths} chemins simulés', linewidth=2, alpha=0.8)
        ax_paths.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
        st.pyplot(fig_paths)
        st.caption(f"Visualisation de différents chemins possibles pour le taux de change {devise_domestique}/{devise_etrangere} sous Garman-Kohlhagen.")

    with col2:
        st.subheader("📉 Convergence du prix MC")
        try:
            plot_convergence(S, K, T, r_d, r_f, sigma, option_type, option_style)
        except Exception as e:
            st.error(str(e))

    # === GRAPHIQUES DE SENSIBILITÉ ===
    st.markdown("<h3 style='text-align: center; margin: 40px 0 20px 0; color: #2E3A59;'>🧭 Graphiques de Sensibilité</h3>", unsafe_allow_html=True)
    fig_all_params, axs = plt.subplots(3, 2, figsize=(16, 12))
    params = ["S", "K", "T", "r_d", "r_f", "sigma"]
    param_labels = [
        "Spot FX (S)", 
        "Strike (K)", 
        "Maturité (T)", 
        "Taux sans risque domestique (r_d)", 
        "Taux sans risque étranger (r_f)", 
        "Volatilité (sigma)"
    ]
    for i, param_key in enumerate(params):
        ax = axs[i // 2, i % 2]
        param_range = None
        if param_key == "S":
            param_range = np.linspace(S * 0.5, S * 1.5, 100)
        elif param_key == "K":
            param_range = np.linspace(K * 0.5, K * 1.5, 100)
        elif param_key == "T":
            param_range = np.linspace(0.01, T * 2, 100)
        elif param_key == "r_d":
            param_range = np.linspace(r_d - 0.05, r_d + 0.05, 100)
        elif param_key == "r_f":
            param_range = np.linspace(r_f - 0.05, r_f + 0.05, 100)
        elif param_key == "sigma":
            param_range = np.linspace(sigma * 0.5, sigma * 1.5, 100)
        prices = []
        for param_value in param_range:
            if param_key == "S":
                prices.append(garman_kohlhagen(param_value, K, T, r_d, r_f, sigma, option_type))
            elif param_key == "K":
                prices.append(garman_kohlhagen(S, param_value, T, r_d, r_f, sigma, option_type))
            elif param_key == "T":
                prices.append(garman_kohlhagen(S, K, param_value, r_d, r_f, sigma, option_type))
            elif param_key == "r_d":
                prices.append(garman_kohlhagen(S, K, T, param_value, r_f, sigma, option_type))
            elif param_key == "r_f":
                prices.append(garman_kohlhagen(S, K, T, r_d, param_value, sigma, option_type))
            elif param_key == "sigma":
                prices.append(garman_kohlhagen(S, K, T, r_d, r_f, param_value, option_type))
        ax.plot(param_range, prices, label=f"Prix en fonction de {param_labels[i]}")
        ax.set_xlabel(param_labels[i])
        ax.set_ylabel("Prix de l'option")
        ax.legend()
        ax.grid(True)
    st.pyplot(fig_all_params)
    st.caption("Graphiques montrant l'impact de chaque paramètre sur le prix de l'option.")
    # ------ BOUTON POUR PRICING DES OPTIONS SUR ACTIONS ------
    if st.button("📊 Voir aussi : Pricing des options sur actions"):
        # Titre du projet
        st.title("Pricing des Options et Visualisations")
        
        # Chemin du fichier PDF
        pdf_path = "C:/Users/HOME/Downloads/projet_couverture_risque_change/Rapport_Théorique.pdf"

        # Personnalisation de l'interface
        st.markdown(
            """
            <style>
            .report-container {
                background-color: #1E1E2F;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
            }
            .title {
                color: #FFD700;
                font-size: 28px;
                text-align: center;
                font-weight: bold;
            }
            .subtitle {
                color: #FFFFFF;
                font-size: 20px;
                font-weight: bold;
            }
            .description {
                color: #CCCCCC;
                font-size: 16px;
            }
            .download-button {
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Interface du rapport
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown('<p class="title">📘 Rapport : Modèles Mathématiques de Valorisation des Options</p>', unsafe_allow_html=True)
        st.markdown('<p class="description">Ce rapport fournit une analyse approfondie des modèles de valorisation des options utilisés en finance :</p>', unsafe_allow_html=True)

        # Liste des modèles avec tirets
        st.markdown("""
        - 📌 **Le modèle de Black-Scholes** : Présentation de la formule et de ses hypothèses.
        - 📌 **Le modèle binomial** : Construction d’un arbre binaire pour évaluer les options.
        - 📌 **Le modèle trinomial** : Extension du modèle binomial avec une branche supplémentaire.
        - 📌 **La méthode de Monte-Carlo** : Simulation aléatoire pour estimer la valeur des options.
        - 📌 Etude de la Sensibilité via l’analyse des différents greeks.           
        """)

        # Bouton de téléchargement stylisé
        st.markdown('<div class="download-button">', unsafe_allow_html=True)
        with open(pdf_path, "rb") as file:
            st.download_button(label="📥 Télécharger le rapport", 
                            data=file, 
                            file_name="Rapport_Valorisation_Options.pdf", 
                            mime="application/pdf")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.write("Ce calculateur vous permet d'évaluer les prix des options et de visualiser les résultats selon différents modèles. Suivez les instructions ci-dessous pour entrer les paramètres nécessaires et obtenir vos résultats. 📈")

        # Section des paramètres
        with st.container():
            st.markdown("<h3 style='text-align: center; color: #2E3A59;'>📌 Paramètres de l'Option</h3>", 
                    unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                S = st.slider("Prix Spot (S)", 0.0, 1000.0, 100.0, 1.0)
                K = st.slider("Strike Price (K)", 0.0, 1000.0, 105.0, 1.0)
                T = st.slider("Maturité (années)", 0.01, 100.0, 1.0, 0.01)
                r = st.slider("Taux Sans Risque (%)", 0.0, 100.0, 5.0, 0.1) / 100
            
            with col2:
                sigma = st.slider("Volatilité (%)", 1.0, 100.0, 20.0, 0.1) / 100
                steps = st.slider("Nombre d'Étapes", 1, 100, 5)
                num_simulations = st.slider("Simulations MC", 1000, 100000, 10000)
                option_type = st.selectbox("Type d'Option", ["Call", "Put"], key="option_type_tab").lower()
                option_style = st.selectbox("Style d'Option", ["Européenne", "Américaine"], key="option_style_tab").lower()

        st.markdown("<hr>", unsafe_allow_html=True)

        # Calcul des résultats
        option_price_bs = black_scholes(S, K, T, r, sigma, option_type)
        option_price_mc = monte_carlo_pricing(S, K, T, r, sigma, option_type, num_simulations)
        prices_binomial, values_binomial = binomial_tree(S, K, T, r, sigma, steps, option_type, option_style)
        prices_trinomial, values_trinomial = trinomial_tree(S, K, T, r, sigma, steps, option_type, option_style)
        delta, gamma, theta, vega, rho = calculate_greeks(S, K, T, r, sigma, option_type)

        # Affichage des résultats
        st.markdown("<h3 style='text-align: center; margin: 30px 0; color: #2E3A59;'>📊 Résultats du Pricing</h3>", 
                unsafe_allow_html=True)

        cols = st.columns(4)
        models = [
            ("Black-Scholes", option_price_bs, "#4e79a7"),
            ("Monte Carlo", option_price_mc, "#f28e2b"),
            ("Binomial", values_binomial[0,0], "#e15759"),
            ("Trinomial", values_trinomial[steps,0], "#59a14f")
        ]

        for (name, value, color), col in zip(models, cols):
            with col:
                st.markdown(f"""
                    <div style='
                        padding: 20px;
                        border-radius: 10px;
                        background: {color}10;
                        border-left: 4px solid {color};
                        margin: 10px 0;
                    '>
                        <h4 style='margin:0; color: {color};'>{name}</h4>
                        <h2 style='margin:5px 0; color: #2E3A59;'>{value:.2f} $</h2>
                    </div>
                    """, unsafe_allow_html=True)

        # Affichage des Grecs
        st.markdown("<h3 style='text-align: center; margin: 40px 0 20px 0; color: #2E3A59;'>📉 Sensibilités Financières</h3>", 
                unsafe_allow_html=True)
        
        greeks = [
            ("Δ Delta", delta, "#8cd17d"),
            ("Γ Gamma", gamma, "#86bcb6"),
            ("Θ Theta", theta, "#e15759"),
            ("ν Vega", vega, "#79706e"),
            ("ρ Rho", rho, "#d7b5a6")
        ]
        
        cols = st.columns(5)
        for (name, value, color), col in zip(greeks, cols):
            with col:
                st.markdown(f"""
                    <div style='
                        padding: 15px;
                        border-radius: 8px;
                        background: {color}10;
                        text-align: center;
                        border: 1px solid {color}30;
                        margin: 5px 0;
                    '>
                        <div style='font-size: 1.2em; color: {color}; font-weight: bold;'>{name}</div>
                        <div style='font-size: 1.4em; color: #2E3A59;'>{value:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Visualisation des arbres
        st.markdown("<h3 style='text-align: center; margin: 40px 0 20px 0; color: #2E3A59;'>🌳 Visualisation des Arbres de Pricing</h3>", 
                unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(10, 6))
            draw_treee(prices_binomial, values_binomial, steps, option_type, 'binomial')
            st.pyplot(plt.gcf())
            st.caption("Arbre Binomial - Chemin optimal en rouge")

        with col2:
            plt.figure(figsize=(10, 6))
            draw_treee(prices_trinomial, values_trinomial, steps, option_type, 'trinomial')
            st.pyplot(plt.gcf())
            st.caption("Arbre Trinomial - Chemins de prix")

        # Graphiques d'évolution
        st.markdown("<h3 style='text-align: center; margin: 40px 0 20px 0; color: #2E3A59;'>📈 Évolution Temporelle des Prix</h3>", 
                unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Comparaison des Modèles", "Simulations Monte Carlo"])
        
        with tab1:
            # Contenu de l'onglet de comparaison des modèles
            fig = plot_evolution(S, K, T, r, sigma, steps, num_simulations)
            st.pyplot(fig)
            
        with tab2:
            # Création de deux colonnes
            col1, col2 = st.columns(2)

            # Première colonne - Simulations Monte Carlo
            with col1:
                st.subheader("🎲 Simulations Monte Carlo")

                dt = T / steps
                num_paths = st.slider("Nombre de chemins à visualiser", 100, 5000, 1000)

                # Création de la figure pour les chemins
                fig_paths = plt.figure(figsize=(10, 6), facecolor='#f0f2f6')
                ax_paths = fig_paths.add_subplot(111)

                # Génération des chemins avec une palette de couleurs rouges
                cmap = plt.get_cmap('Reds')
                for i in range(num_paths):
                    path = np.zeros(steps + 1)
                    path[0] = S
                    for j in range(1, steps + 1):
                        path[j] = path[j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
                    ax_paths.plot(path, color=cmap(i / num_paths), alpha=0.7, linewidth=0.9)

                # Amélioration du style
                ax_paths.set_title("Simulations de Monte Carlo - Chemins de prix", fontsize=16, pad=20, color="#333333")
                ax_paths.set_xlabel("Étapes", fontsize=14, labelpad=10, color="#333333")
                ax_paths.set_ylabel("Prix de l'actif", fontsize=14, labelpad=10, color="#333333")
                ax_paths.grid(True, linestyle='--', alpha=0.5, color="#cccccc")
                ax_paths.set_facecolor('#f8f9fa')

                # Ajout d'une légende dynamique
                ax_paths.plot([], [], color='gray', label=f'{num_paths} chemins simulés', linewidth=2, alpha=0.8)
                ax_paths.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)

                st.pyplot(fig_paths)

                # Ajout d'une note explicative
                st.caption("Visualisation de différents chemins que pourrait prendre le prix de l'actif sous-jacent selon le modèle de Black-Scholes.")

            # Deuxième colonne - Convergence
            with col2:
                st.subheader("📉 Convergence du prix")

                # Ajout d'un curseur pour le nombre de points de convergence
                convergence_points = st.slider("Nombre de points de convergence", 20, 200, 50)

                sample_sizes = np.linspace(100, num_simulations, convergence_points, dtype=int)
                means = []
                stds = []

                # Barre de progression
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, n in enumerate(sample_sizes):
                    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.randn(n))
                    payoffs = np.maximum(ST - K, 0)
                    means.append(np.exp(-r * T) * np.mean(payoffs))
                    stds.append(np.exp(-r * T) * np.std(payoffs) / np.sqrt(n))
                    progress_bar.progress((idx + 1) / len(sample_sizes))
                    status_text.text(f"Calcul en cours... {idx + 1}/{len(sample_sizes)}")

                bs_price = black_scholes(S, K, T, r, sigma, 'call')

                # Création de la figure pour la convergence
                fig_conv = plt.figure(figsize=(10, 6), facecolor='#f0f2f6')
                ax_conv = fig_conv.add_subplot(111)

                # Amélioration du tracé de convergence
                ax_conv.plot(sample_sizes, means, label='Prix estimé', color='#d62728', linewidth=2.5, alpha=0.9)
                ax_conv.fill_between(sample_sizes, 
                                    np.array(means) - 1.96 * np.array(stds),
                                    np.array(means) + 1.96 * np.array(stds), 
                                    color='#d62728', alpha=0.2, label='Intervalle de confiance (95%)')
                ax_conv.axhline(bs_price, color='#7f0e0e', linestyle='--', linewidth=2, label='Black-Scholes')

                # Style amélioré
                ax_conv.set_title("Convergence du prix estimé vers Black-Scholes", fontsize=16, pad=20, color="#333333")
                ax_conv.set_xlabel("Nombre de simulations", fontsize=14, labelpad=10, color="#333333")
                ax_conv.set_ylabel("Prix de l'option", fontsize=14, labelpad=10, color="#333333")
                ax_conv.grid(True, linestyle='--', alpha=0.5, color="#cccccc")
                ax_conv.set_facecolor('#f8f9fa')
                ax_conv.legend(loc='upper right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)

                # Ajout d'une annotation pour le prix final
                final_price = means[-1]
                ax_conv.annotate(f'Prix final: {final_price:.2f}', 
                                xy=(sample_sizes[-1], final_price),
                                xytext=(-80, 10), textcoords='offset points',
                                fontsize=12, color="#333333",
                                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                                arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6))

                st.pyplot(fig_conv)
                progress_bar.empty()
                status_text.empty()

                # Affichage des valeurs numériques
                st.metric(label="Prix Black-Scholes", value=f"{bs_price:.2f}")
                st.metric(label="Dernière estimation Monte Carlo", value=f"{means[-1]:.2f}")

                # Calcul de l'erreur relative
                error = abs(means[-1] - bs_price) / bs_price * 100
                st.metric(label="Erreur relative", value=f"{error:.2f}%")
import numpy as np
with tab11:
    import streamlit as st
    import numpy as np
    import pandas as pd
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    st.title("Scénarios de Stress FX ")

    # ----------- Encadrement Neon pour la description -----------
    st.markdown("""
    <style>
    .neon-desc {
        background: rgba(15, 15, 30, 0.92);
        border: 2px solid rgba(0, 255, 255, 0.7);
        border-radius: 20px;
        padding: 22px 30px;
        margin-bottom: 30px;
        margin-top: 10px;
        box-shadow: 0 0 18px 2px #00ffff55, 0 0 45px 10px #00ffff22 inset;
        color: white;
        font-size: 17px;
        line-height: 1.65;
        backdrop-filter: blur(5px);
    }
    </style>
    <div class="neon-desc">
    Ce module permet de <b>simuler des scénarios de crise sur le MAD, l'EUR, et le USD</b>.<br>
    Vous pouvez visualiser l'impact de chocs extrêmes sur la valeur d'un portefeuille et explorer plusieurs situations via des graphiques pédagogiques et des statistiques de risque.<br><br>
    <b>Fonctionnalités essentielles :</b>
    <ul>
        <li>Réglage des expositions (MAD, EUR, USD) et des taux de change</li>
        <li>Choix des scénarios de stress (dévaluation MAD, appréciation EUR/USD)</li>
        <li>Décomposition graphique (waterfall) de l'impact de chaque choc</li>
        <li>Sensibilité à la dévaluation du MAD (graphique interactif)</li>
        <li>Tableau multi-scénarios et export CSV</li>
        <li>Simulation aléatoire de scénarios : distribution des pertes/gains, VaR, percentiles</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # === Paramètres du portefeuille ===
    st.header("1. Paramètres du portefeuille")
    col1, col2, col3 = st.columns(3)
    with col1:
        exposition_mad = st.slider(
            "Exposition MAD", min_value=0, max_value=50_000_000, value=10_000_000, step=100_000
        )
    with col2:
        exposition_eur = st.slider(
            "Exposition EUR", min_value=0, max_value=5_000_000, value=1_000_000, step=50_000
        )
    with col3:
        exposition_usd = st.slider(
            "Exposition USD", min_value=0, max_value=5_000_000, value=500_000, step=50_000
        )

    col4, col5 = st.columns(2)
    with col4:
        spot_mad_eur = st.slider(
            "Taux MAD/EUR", min_value=10.0, max_value=12.0, value=11.0, step=0.01
        )
    with col5:
        spot_mad_usd = st.slider(
            "Taux MAD/USD", min_value=9.0, max_value=12.0, value=10.1, step=0.01
        )

    # ==== SCÉNARIO PRINCIPAL ====
    st.header("2. Configuration du Scénario Principal")
    col1, col2, col3 = st.columns(3)
    with col1:
        mad_deval = st.slider("Dévaluation du MAD (%)", min_value=-50, max_value=0, value=-20, step=1)
    with col2:
        eur_app = st.slider("Appréciation EUR (%)", min_value=0, max_value=50, value=10, step=1)
    with col3:
        usd_app = st.slider("Appréciation USD (%)", min_value=0, max_value=50, value=10, step=1)

    def portefeuille_value(mad, eur, usd, taux_mad_eur, taux_mad_usd):
        # Conversion en MAD
        return mad + eur * taux_mad_eur + usd * taux_mad_usd

    # Calcul des taux après stress
    mad_eur_stress = spot_mad_eur * (1 + eur_app/100) / (1 + abs(mad_deval)/100)
    mad_usd_stress = spot_mad_usd * (1 + usd_app/100) / (1 + abs(mad_deval)/100)

    # Valeur du portefeuille
    valeur_init = portefeuille_value(exposition_mad, exposition_eur, exposition_usd, spot_mad_eur, spot_mad_usd)
    valeur_stress = portefeuille_value(exposition_mad, exposition_eur, exposition_usd, mad_eur_stress, mad_usd_stress)
    perte = valeur_stress - valeur_init
    perte_pct = perte / valeur_init * 100 if valeur_init else 0

    st.markdown("### Résumé du scénario principal")
    st.markdown(f"""
    - **Valeur initiale du portefeuille (MAD)** : {valeur_init:,.2f}  
    - **Valeur après stress (MAD)** : {valeur_stress:,.2f}  
    - **Impact du stress (perte/gain)** : {perte:,.2f}  (**{perte_pct:.2f}%**)
    """)

    st.markdown("#### Taux de change après stress")
    st.markdown(f"""
    - **MAD/EUR stressé** : {mad_eur_stress:.4f}  
    - **MAD/USD stressé** : {mad_usd_stress:.4f}
    """)

    # === Waterfall : impact de chaque choc ===
    waterfall_data = [
        {'label': 'Valeur Initiale', 'value': valeur_init},
        {'label': 'Dévaluation MAD', 'value': portefeuille_value(exposition_mad, exposition_eur, exposition_usd,
                                                                 spot_mad_eur / (1 + abs(mad_deval)/100),
                                                                 spot_mad_usd / (1 + abs(mad_deval)/100)) - valeur_init},
        {'label': 'Appréciation EUR', 'value': portefeuille_value(0, exposition_eur, 0,
                                                                 spot_mad_eur * (1 + eur_app/100) / (1 + abs(mad_deval)/100), 0)
                                                - portefeuille_value(0, exposition_eur, 0,
                                                                 spot_mad_eur / (1 + abs(mad_deval)/100), 0)},
        {'label': 'Appréciation USD', 'value': portefeuille_value(0, 0, exposition_usd, 0,
                                                                 spot_mad_usd * (1 + usd_app/100) / (1 + abs(mad_deval)/100))
                                                - portefeuille_value(0, 0, exposition_usd, 0,
                                                                 spot_mad_usd / (1 + abs(mad_deval)/100))},
        {'label': 'Valeur finale', 'value': valeur_stress}
    ]

    # Calcul cumulé pour waterfall
    waterfall_y = [waterfall_data[0]['value']]
    for d in waterfall_data[1:-1]:
        waterfall_y.append(waterfall_y[-1] + d['value'])
    waterfall_y.append(waterfall_data[-1]['value'])

    waterfall_labels = [d['label'] for d in waterfall_data]
    waterfall_deltas = [0] + [d['value'] for d in waterfall_data[1:-1]] + [waterfall_y[-1] - waterfall_y[-2]]

    fig_waterfall = go.Figure(go.Waterfall(
        x=waterfall_labels,
        y=waterfall_deltas,
        base=valeur_init,
        measure=["absolute", "relative", "relative", "relative", "total"],
        text=[f"{v:,.0f}" for v in waterfall_y],
        decreasing={"marker":{"color":"#d62728"}},
        increasing={"marker":{"color":"#2ca02c"}},
        totals={"marker":{"color":"#1f77b4"}},
    ))
    fig_waterfall.update_layout(title="Waterfall : Détail de l'impact des chocs", showlegend=False)
    st.plotly_chart(fig_waterfall, use_container_width=True)
    st.caption("Ce graphique montre l’effet étape par étape de chaque choc (dévaluation du MAD, appréciation EUR/USD) sur votre portefeuille.")

    # === GRAPHIQUE 2D : Impact de la dévaluation du MAD ===
    st.markdown("### Impact de la dévaluation du MAD")
    mad_devals = np.linspace(-40, 0, 100)
    values = []
    for m in mad_devals:
        mad_eur = spot_mad_eur * (1 + eur_app/100) / (1 + abs(m)/100)
        mad_usd = spot_mad_usd * (1 + usd_app/100) / (1 + abs(m)/100)
        v = portefeuille_value(exposition_mad, exposition_eur, exposition_usd, mad_eur, mad_usd)
        values.append(v)

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=mad_devals, y=values, mode='lines', name="Valeur Portefeuille"
    ))
    fig_line.add_hline(y=valeur_init, line_dash="dot", annotation_text="Valeur Initiale", line_color="green")
    fig_line.update_layout(
        title="Evolution de la valeur du portefeuille selon la dévaluation du MAD",
        xaxis_title="Dévaluation du MAD (%)",
        yaxis_title="Valeur du portefeuille (MAD)",
        template="plotly_white"
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.caption(
        "Ce graphique montre comment la valeur du portefeuille évolue en fonction de la dévaluation du MAD, "
        "en gardant la configuration du scénario (appréciation EUR/USD) constante."
    )

    # === GRILLE DES SCÉNARIOS ===
    st.markdown("### Tableau interactif de scénarios multiples")
    mad_devals_grid = np.arange(-40, 1, 5)
    eur_app_grid = np.arange(0, 21, 10)
    usd_app_grid = np.arange(0, 21, 10)
    table_data = []
    for m in mad_devals_grid:
        for e in eur_app_grid:
            for u in usd_app_grid:
                mad_eur = spot_mad_eur * (1 + e/100) / (1 + abs(m)/100)
                mad_usd = spot_mad_usd * (1 + u/100) / (1 + abs(m)/100)
                v = portefeuille_value(exposition_mad, exposition_eur, exposition_usd, mad_eur, mad_usd)
                table_data.append({
                    "Déval MAD (%)": m,
                    "EUR App (%)": e,
                    "USD App (%)": u,
                    "Valeur Portefeuille (MAD)": int(v)
                })

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table.style.background_gradient(cmap='RdYlGn_r', subset=["Valeur Portefeuille (MAD)"]))
    st.download_button("📥 Exporter la grille de scénarios (CSV)", df_table.to_csv(index=False), file_name="stress_grid.csv")
    st.caption("Ce tableau récapitule l’impact de différentes combinaisons de chocs sur la valeur du portefeuille.")

    # === SIMULATION DISTRIBUTIONNELLE : "Tirage de Scénarios" ===
    st.markdown("### Simulateur de scénarios aléatoires")
    st.markdown(
        "Tirez au sort 1000 scénarios de crises potentielles pour visualiser la distribution des pertes potentielles."
    )
    n_sim = st.slider("Nombre de scénarios simulés", 500, 5000, 1000, 500)
    mad_deval_sim = np.random.uniform(-40, 0, n_sim)
    eur_app_sim = np.random.uniform(0, 20, n_sim)
    usd_app_sim = np.random.uniform(0, 20, n_sim)
    sim_values = []
    for m, e, u in zip(mad_deval_sim, eur_app_sim, usd_app_sim):
        mad_eur = spot_mad_eur * (1 + e/100) / (1 + abs(m)/100)
        mad_usd = spot_mad_usd * (1 + u/100) / (1 + abs(m)/100)
        v = portefeuille_value(exposition_mad, exposition_eur, exposition_usd, mad_eur, mad_usd)
        sim_values.append(v)
    sim_values = np.array(sim_values)
    losses = sim_values - valeur_init

    VaR_95 = np.percentile(losses, 5)
    median = np.percentile(losses, 50)
    worst = losses.min()
    best = losses.max()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=losses, nbinsx=40, marker_color='#d62728', name="Distribution des pertes/gains"
    ))
    fig_hist.add_vline(x=perte, line_dash="dash", line_color="blue", annotation_text="Scénario Principal")
    fig_hist.update_layout(
        title="Distribution des pertes/gains sur les scénarios simulés",
        xaxis_title="Pertes/Gains (MAD)",
        yaxis_title="Nombre de scénarios",
        template="plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(f"""
    **Statistiques de pertes simulées :**
    - Pire scénario : {worst:,.0f} MAD
    - VaR 95% (perte sur les 5% pires cas) : {VaR_95:,.0f} MAD
    - Perte médiane : {median:,.0f} MAD
    - Meilleur scénario : {best:,.0f} MAD
    """)

    st.caption(
        "Ce graphique montre la distribution des pertes (ou gains) sur un grand nombre de scénarios tirés au hasard. "
        "La ligne bleue correspond au scénario principal configuré plus haut."
    )

    # === EXPLICATIONS PEDAGOGIQUES ===
    with st.expander("Comment interpréter ces résultats ? (cliquez ici)"):
        st.markdown("""
        - <b>Waterfall</b> : vous voyez l'impact de chaque choc séparément sur la valeur de votre portefeuille.
        - <b>Evolution 2D</b> : vous comprenez la sensibilité de votre portefeuille à la seule dévaluation du MAD.
        - <b>Tableau</b> : vous pouvez comparer facilement plusieurs combinaisons de chocs.
        - <b>Distribution</b> : vous voyez la probabilité de réaliser une perte/un gain sur un grand nombre de scénarios de stress.
        - <b>Interprétation :</b>
            <ul>
            <li>Plus la courbe est à gauche, plus le portefeuille est vulnérable à la crise.</li>
            <li>Si le scénario principal (ligne bleue) est dans la queue gauche, il s'agit d'un scénario rare mais dangereux.</li>
            <li>La VaR 95% représente la perte maximale sur 95% des cas simulés.</li>
            </ul>
        """, unsafe_allow_html=True)
with tab10:
    import numpy as np
    import pandas as pd
    from datetime import date, datetime, timedelta
    from pandas.tseries.offsets import DateOffset
    from scipy import interpolate
    from dateutil.relativedelta import relativedelta
    import streamlit as st
    import plotly.graph_objects as go
    from streamlit_lottie import st_lottie

    st.title("Currency Swap Pricing Tool")

    # ----------- Encadré d'utilité du module (affichage direct) -----------
    st.markdown("""
    <div class="intro-box">
        <b>À quoi sert ce module&nbsp;?</b><br>
        <ul>
            <li>Estimez la juste valeur et la sensibilité (<b>DV01</b>) d'un swap de devises EUR/MAD ou USD/MAD selon vos paramètres.</li>
            <li>Simulez les cash-flows fixes et flottants pour chaque jambe du swap sur toute la durée de vie du produit.</li>
            <li>Importez vos propres courbes de taux et comparez différents scénarios de couverture ou de financement.</li>
            <li>Visualisez l'impact des variations de taux et de change sur la valorisation du swap.</li>
        </ul>
        <i>Ce module est idéal pour la gestion du risque, l’optimisation financière, ou la formation à la pratique du swap de devises.</i>
    </div>
    """, unsafe_allow_html=True)

    # ----------- Description pédagogique déroulante (expander) -----------
    with st.expander("Afficher la description détaillée du module et la pédagogie"):
        st.markdown("""
        <div class="intro-box">
        <b>Qu'est-ce qu'un Swap de Devises ?</b><br>
        <br>
        Un <b>swap de devises</b> (ou "currency swap") est un produit dérivé qui permet à deux parties d'échanger des flux d'intérêts et/ou de principal dans deux devises différentes, selon un calendrier et des modalités convenus à l'avance.<br>
        <br>
        <ul>
            <li>À la date de départ, les deux parties échangent les montants notionnels dans leurs devises respectives au taux de change du marché (spot ou forward).</li>
            <li>Pendant la durée du swap, chaque partie verse à l'autre les intérêts calculés sur le nominal dans sa propre devise (flux fixes ou flottants, selon la convention).</li>
            <li>À la date de maturité, les deux parties se rééchangent les montants notionnels aux conditions initiales.</li>
        </ul>
        <b>Utilités principales :</b>
        <ul>
            <li>Se couvrir contre le risque de change et de taux d'intérêt simultanément.</li>
            <li>Accéder à des financements ou placements dans une devise étrangère à des conditions optimisées.</li>
            <li>Adapter la structure de ses flux financiers à ses besoins opérationnels.</li>
        </ul>
        <b>Exemple :</b><br>
        Une entreprise marocaine souhaite emprunter en EUR mais dispose de cash-flows en MAD. Un swap EUR/MAD lui permet d'emprunter en MAD, puis d'échanger les flux d'intérêts et de principal contre ceux d'un emprunt EUR, tout en neutralisant le risque de change.
        </div>
        """, unsafe_allow_html=True)

   
    def refreshCurve(data, currency):
        """
        Interpolate yield curves for a given currency.
        """
        dataRatesCurve = data.to_numpy()
        rates = dataRatesCurve[:, 1:]
        time = np.array(list(range(1, 21)))
        startDate = date.today()
        year = int(startDate.strftime("%Y"))
        endDate = startDate.replace(year=year + 50)
        delta = endDate - startDate
        deltaDays = delta.days
        timeNew = np.linspace(1, 20, deltaDays)

        interpolated_curves = {}
        for i, rate_type in enumerate(["All", "Ester", "Eurib1", "Eurib3", "Eurib6", "Eurib12"]):
            interpo = interpolate.interp1d(time, rates[:, i])
            interpolated_curves[rate_type] = interpo(timeNew)

        return pd.DataFrame(interpolated_curves, index=pd.date_range(start=startDate, periods=deltaDays, freq='D'))


    def currencySwapComputation(
        startDate,
        endDate,
        payOrRec,
        notionalEUR,
        notionalUSD,
        notionalMAD,
        rateFixed,
        floatIndex,
        setfFixed,
        setfFloat,
        basFixed,
        basFloat,
        dataRatesCurveEUR,
        dataRatesCurveUSD,
        dataRatesCurveMAD,
        fxSpotEURMAD,
        fxSpotUSDMAD,
        fxForwardEURMAD,
        fxForwardUSDMAD,
    ):
        """
        Compute the price and DV01 for a currency swap (EUR/MAD and USD/MAD).
        """
        startDate = datetime.strptime(startDate, '%m/%d/%y')
        endDate = datetime.strptime(endDate, '%m/%d/%y')

        # Frequency of fixed leg
        if setfFixed == "Annual":
            periodFixed = 12
        elif setfFixed == "Semi-Annual":
            periodFixed = 6
        elif setfFixed == "Quarterly":
            periodFixed = 3
        elif setfFixed == "Monthly":
            periodFixed = 1

        # Frequency of floating leg
        if setfFloat == "Annual":
            periodFloat = 12
        elif setfFloat == "Semi-Annual":
            periodFloat = 6
        elif setfFloat == "Quarterly":
            periodFloat = 3
        elif setfFloat == "Monthly":
            periodFloat = 1

        # Base of fixed leg
        if basFixed == "A360":
            base1 = 360
        elif basFixed == "A365":
            base1 = 365

        # Base of floating leg
        if basFloat == "A360":
            base2 = 360
        elif basFloat == "A365":
            base2 = 365

        # Pay or Receive fixed leg
        payrec = -1 if payOrRec == 1 else 1

        # Fixed Leg (MAD)
        months = (endDate.year - startDate.year) * 12 + endDate.month - startDate.month
        datesRollFixed = pd.date_range(start=startDate, periods=months // periodFixed + 1, freq=f'{periodFixed}M')
        nbrDays = (datesRollFixed[1:] - datesRollFixed[:-1]).days
        flowsFixed = payrec * notionalMAD * rateFixed / 100 * nbrDays / base1

        # Floating Leg (EUR or USD)
        if floatIndex in dataRatesCurveUSD.columns:
            dataRatesCurveFloat = dataRatesCurveUSD[floatIndex]
            fxSpot = fxSpotUSDMAD
            fxForward = fxForwardUSDMAD
            notionalFloat = notionalUSD
        elif floatIndex in dataRatesCurveEUR.columns:
            dataRatesCurveFloat = dataRatesCurveEUR[floatIndex]
            fxSpot = fxSpotEURMAD
            fxForward = fxForwardEURMAD
            notionalFloat = notionalEUR
        else:
            raise ValueError("Invalid float index")

        datesRollFloat = pd.date_range(start=startDate, periods=months // periodFloat + 1, freq=f'{periodFloat}M')
        nbrDaysFloat = (datesRollFloat[1:] - datesRollFloat[:-1]).days
        forwardRates = dataRatesCurveFloat.loc[datesRollFloat[:-1]].values
        flowsFloat = -payrec * notionalFloat * forwardRates * fxForward / fxSpot * nbrDaysFloat / base2

        # Discounted Cash Flows
        discountFactors = dataRatesCurveMAD["All"].loc[datesRollFixed[1:]].values
        discountedFixedLeg = np.sum(flowsFixed / (1 + discountFactors / 100) ** (nbrDays / base1))
        discountedFloatLeg = np.sum(flowsFloat / (1 + discountFactors / 100) ** (nbrDaysFloat / base2))

        # Price of the swap
        priceSwap = discountedFixedLeg + discountedFloatLeg

        # DV01
        bumpedForwardRates = forwardRates + 0.01
        bumpedFlowsFloat = -payrec * notionalFloat * bumpedForwardRates * fxForward / fxSpot * nbrDaysFloat / base2
        bumpedDiscountedFloatLeg = np.sum(bumpedFlowsFloat / (1 + discountFactors / 100) ** (nbrDaysFloat / base2))
        dv01 = bumpedDiscountedFloatLeg - discountedFloatLeg

        return priceSwap, dv01, discountedFixedLeg, discountedFloatLeg

    # Custom CSS for premium design
   
    


    # Tabs for Navigation
    tab1, tab2, tab3 = st.tabs(["Paramètres", "Courbes de Taux", "Résultats"])

    # Tab 1: Paramètres
    with tab1:
        st.subheader("🔧 Paramètres du Swap")
        col1, col2 = st.columns(2)
        with col1:
            swap_type = st.selectbox("Type de Swap", ["EUR/MAD", "USD/MAD"])
            notional = st.slider("Notional (en devise étrangère)", min_value=1.0, max_value=10_000_000.0, value=1_000_000.0, step=100_000.0)
            fixed_rate = st.slider("Taux Fixe (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.01)
        with col2:
            pay_or_receive = st.radio("Flux Fixes", ["Payer", "Recevoir"], horizontal=True)
            start_date = st.date_input("Date de début", value=date.today())
            end_date = st.date_input("Date de fin", value=date.today() + timedelta(days=365))

        st.subheader("🔹 Paramètres de la Jambe Flottante")
        col3, col4 = st.columns(2)
        with col3:
            float_index = st.selectbox(
                "Index Flottant (Indice de Référence)",
                options=["Euribor 3M", "Euribor 6M", "USD Libor 3M", "USD Libor 6M"],
            )
            float_spread = st.slider("Spread Flottant (bps)", min_value=0, max_value=500, value=50, step=1)
        with col4:
            fixed_leg_frequency = st.selectbox("Fréquence des Flux Fixes", ["Annuel", "Semestriel", "Trimestriel"])
            float_leg_frequency = st.selectbox("Fréquence des Flux Flottants", ["Annuel", "Semestriel", "Trimestriel"])

    # Tab 2: Upload Courbes de Taux
    with tab2:
        st.subheader("📂 Téléchargez les Courbes de Taux")
        uploaded_file = st.file_uploader("Chargez un fichier CSV (Courbes de Taux)", type=["csv"])
        if uploaded_file:
            curves_data = pd.read_csv(uploaded_file)
            st.success("Courbes de taux chargées avec succès !")
            st.dataframe(curves_data.head(10))
        else:
            st.warning("Veuillez télécharger un fichier CSV contenant les courbes de taux.")

    # Tab 3: Résultats
    with tab3:
        st.subheader("📊 Résultats du Calcul")
        if st.button("Lancer le Calcul"):
            if not uploaded_file:
                st.error("Veuillez fournir les courbes de taux pour effectuer le calcul.")
            else:
                # Placeholder for calculation
                price_swap = np.random.uniform(-50000, 50000)  # Replace with actual computation
                dv01 = np.random.uniform(-100, 100)  # Replace with actual computation

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f'<div class="metric-card">💰 Prix du Swap : <br><b>{price_swap:,.2f} MAD</b></div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f'<div class="metric-card">📈 DV01 : <br><b>{dv01:,.2f} MAD</b></div>',
                        unsafe_allow_html=True,
                    )

                # Placeholder for cash flow data
                dates = pd.date_range(start=start_date, periods=10, freq="6M")
                cash_flow_fixed = pd.DataFrame(
                    {"Date": dates, "Flux Fixes (MAD)": np.random.uniform(10000, 20000, 10)}
                )
                cash_flow_float = pd.DataFrame(
                    {"Date": dates, "Flux Flottants (MAD)": np.random.uniform(-15000, -5000, 10)}
                )

                st.write("### 🔹 Flux de la Jambe Fixe")
                st.dataframe(cash_flow_fixed)

                st.write("### 🔹 Flux de la Jambe Flottante")
                st.dataframe(cash_flow_float)

                st.write("### 🔹 Graphique des Flux de Trésorerie")
                st.write("Le graphique ci-dessous montre les flux fixes et flottants au fil du temps :")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cash_flow_fixed["Date"], y=cash_flow_fixed["Flux Fixes (MAD)"], mode="lines+markers", name="Fixe"))
                fig.add_trace(go.Scatter(x=cash_flow_float["Date"], y=cash_flow_float["Flux Flottants (MAD)"], mode="lines+markers", name="Flottant"))
                fig.update_layout(title="Flux de Trésorerie : Fixe vs Flottant", xaxis_title="Date", yaxis_title="Flux (MAD)")
                st.plotly_chart(fig)

with tab7:
    
    import streamlit as st
    import numpy as np
    import plotly.graph_objects as go
    st.title("FX - Options Strategy")
    # =============== ENCADRÉ EXPLICATIF EN HAUT DE MODULE ===============
    st.markdown("""
    <div class="neon-box-fixed">
        <b>Ce module vous permet d'analyser et de comparer différentes stratégies optionnelles sur le marché des changes :</b>
        <ul style="margin-left: -1.5em;">
            <li>Simulez le payoff et le risque de stratégies populaires : <b>Short/Long Call</b>, <b>Short/Long Put</b>, <b>Straddle</b>, <b>Strangle</b>, <b>Spreads</b>…</li>
            <li>Ajustez les paramètres de marché (spot, taux, volatilité, maturité, strikes…)</li>
            <li>Visualisez et comprenez les profils de gain/perte à maturité</li>
            <li>Identifiez les scénarios adaptés à vos besoins de couverture ou de spéculation</li>
        </ul>
        <i>Idéal pour tester l'impact des stratégies optionnelles sur le risque de change, que vous soyez trésorier, étudiant ou investisseur.</i>
    </div>
    """, unsafe_allow_html=True)

    # =============== PARAMÈTRES COMMUNS ================
    def fx_common_inputs():
        c1, c2, c3 = st.columns(3)
        with c1:
            spot = st.number_input("Spot FX (MAD/Devise)", min_value=0.0001, value=11.0, step=0.0001, format="%.4f", key="fx_spot")
        with c2:
            currency = st.selectbox("Devise", options=["EUR", "USD"], index=0, key="fx_currency")
        with c3:
            maturity_days = st.number_input("Maturité (jours)", min_value=1, value=365, step=1, key="fx_maturity")

        c4, c5 = st.columns(2)
        with c4:
            r_d = st.number_input(f"Taux d'intérêt MAD (%)", min_value=0.0, value=3.0, step=0.01, key="fx_r_d")
        with c5:
            r_f = st.number_input(f"Taux d'intérêt {currency} (%)", min_value=0.0, value=2.0, step=0.01, key="fx_r_f")

        sigma = st.slider("Volatilité implicite (%)", min_value=0.1, max_value=150.0, value=15.0, step=0.1, key="fx_sigma")

        # Conversion
        r_d_ = r_d / 100
        r_f_ = r_f / 100
        t_ = maturity_days / 365
        sigma_ = sigma / 100

        return {
            "spot": spot,
            "currency": currency,
            "maturity_days": maturity_days,
            "r_d": r_d,
            "r_f": r_f,
            "sigma": sigma,
            "r_d_": r_d_,
            "r_f_": r_f_,
            "t_": t_,
            "sigma_": sigma_
        }

    # =============== CSS GLOBAL POUR LES ENCADRÉS ET TABLES ===============
    st.markdown("""
    <style>
    body, .reportview-container, .main, .block-container {
        font-size: 22px !important;
        font-family: 'Segoe UI', 'Arial', sans-serif !important;
    }
    .neon-box-fixed {
        background-color: #0f0f0f;
        border-left: 7px solid #00FFFF;
        padding: 1.7rem 1.5rem;
        margin-top: 1.7rem;
        margin-bottom: 1.7rem;
        border-radius: 14px;
        box-shadow: 0 0 18px #00FFFF;
        color: white;
        font-weight: bold;
        text-align: left;
        font-size: 1.28em;
        line-height: 1.7em;
    }
    .fx-plot-neon {
        border: 4px solid #00FFFF;
        border-radius: 18px;
        box-shadow: 0 0 28px #00FFFF;
        padding: 16px 8px 26px 8px;
        background: #101010;
        margin-bottom: 2.5rem;
        margin-top: 2.0rem;
        width: 98%;
        margin-left: auto;
        margin-right: auto;
    }
    .center-table-container {
        display: flex;
        justify-content: center;
        margin-bottom: 2.5rem;
    }
    .result-table {
        margin: auto;
        font-size: 1.19em;
        min-width: 420px;
        max-width: 98vw;
        padding: 1.5em 2em;
    }
    .result-table td {
        font-size: 1.13em; font-weight: bold; padding: 0.44em 1.1em;
    }
    .strategy-btn button {
        font-size: 1.35em;
        font-weight: bold;
        padding: 1.1em 2.2em;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg,#00beff 0,#00ffe7 100%);
        color: #003049;
        cursor: pointer;
        box-shadow: 0 2px 14px #00bfff44;
        transition: 0.13s;
        margin-bottom: 0.5em;
    }
    .strategy-btn button.selected {
        background: linear-gradient(90deg,#003049 0,#003049 100%);
        color: #00ffe7;
        outline: 3px solid #00ffe7;
    }
    .stSlider > div {
        padding-top: 18px !important;
        padding-bottom: 18px !important;
    }
    .stNumberInput > div {
        padding-top: 18px !important;
        padding-bottom: 18px !important;
    }
    .stSelectbox > div {
        padding-top: 18px !important;
        padding-bottom: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # =============== SÉLECTEUR DE STRATÉGIE ===============
    strategies = [
        ("Short Call/Put", "short"),
        ("Long Call/Put", "long"),
        ("Call/Put Spread", "spread"),
        ("Straddle", "straddle"),
        ("Strangle", "strangle"),
    ]
    if "fx_strategy" not in st.session_state:
        st.session_state["fx_strategy"] = "short"

    st.markdown("<div class='strategy-btn' style='display:flex;flex-direction:row;gap:1.5rem;justify-content:center;margin-bottom:2rem;'>", unsafe_allow_html=True)
    cols = st.columns(len(strategies))
    for i, (label, key) in enumerate(strategies):
        if cols[i].button(label, key=f"fx_{key}_btn"):
            st.session_state["fx_strategy"] = key
    st.markdown("</div>", unsafe_allow_html=True)

    # =============== PARAMÈTRES COMMUNS (En haut, une seule fois) ===============
    params = fx_common_inputs()

    # =============== SHORT ===============
    if st.session_state["fx_strategy"] == "short":
        st.markdown("""
        <div class="neon-box-fixed">
        <b>Principe :</b><br>
        La <b>vente d’un call ou d’un put</b> (“short call/put”, ou “vente nue”) consiste à vendre des options call ou put sur le marché des changes.<br>
        Le gain maximum est limité à la prime reçue ; la perte potentielle peut être très importante si l’option est exercée loin dans la monnaie.<br>
        <b>Cette stratégie présente donc un profil de risque élevé, à réserver aux opérateurs expérimentés et couverts.</b>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="neon-box-fixed"> Choisissez <b>Short Call</b> ou <b>Short Put</b></div>
        <div class="neon-box-fixed"> Paramétrez la prime, le strike, la quantité</div>
        <div class="neon-box-fixed"> Visualisez le payoff, le P&L dynamique et les chiffres clés ci-dessous</div>
        """, unsafe_allow_html=True)
        option_type = st.selectbox("Stratégie", ["Call", "Put"], format_func=lambda x: f"Vente {x} (Short {x})", key="short_strat_type")
        premium = st.number_input("Prime reçue (MAD)", min_value=0.01, value=1.0, step=0.01, key="short_premium")
        strike = st.number_input("Strike (MAD/Devise)", min_value=0.0001, value=params["spot"], step=0.0001, format="%.4f", key="short_strike")
        qty = st.number_input("Notionnel (en devise)", min_value=1.0, value=100000.0, step=1.0, format="%.0f", key="short_qty")

        st.markdown("#### Simulation dynamique")
        col_spot, col_mat, col_sigma = st.columns(3)
        with col_spot:
            spot_sim = st.slider("Spot FX simulé", min_value=0.0001, max_value=float(params["spot"]*4), value=float(params["spot"]), step=0.0001, format="%.4f", key="short_spot_sim")
        with col_mat:
            mat_sim = st.slider("Maturité simulée (jours)", min_value=1, max_value=int(params["maturity_days"]), value=int(params["maturity_days"]), step=1, key="short_mat_sim")
        with col_sigma:
            sigma_sim = st.slider("Volatilité simulée (%)", min_value=0.1, max_value=150.0, value=params["sigma"], step=0.1, key="short_sigma_sim")
        sigma_sim_ = sigma_sim / 100
        t_sim_ = mat_sim / 365

        if params["spot"] >= 50:
            stepSize = 0.5
        elif params["spot"] >= 10:
            stepSize = 0.1
        else:
            stepSize = 0.01
        x_spot = np.arange(max(0.0001, params["spot"] * 0.2), params["spot"] * 2 + stepSize, stepSize)
        direction = 1 if option_type == "Call" else -1
        payoff_maturity = [min(premium * qty, premium * qty - qty * direction * (xi - strike)) for xi in x_spot]
        option_current_prices = [garman_kohlhagen(xi, strike, t_sim_, params["r_d_"], params["r_f_"], sigma_sim_, option_type.lower()) for xi in x_spot]
        pnl_current = [(premium - opc) * qty for opc in option_current_prices]
        option_val_sim = garman_kohlhagen(spot_sim, strike, t_sim_, params["r_d_"], params["r_f_"], sigma_sim_, option_type.lower())
        profit = (premium - option_val_sim) * qty
        current_color = "green" if profit >= 0.01 else "red" if profit <= -0.01 else "black"

        st.markdown('<div class="fx-plot-neon">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_spot, y=payoff_maturity, mode="lines", name="Profit & Loss à maturité", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=x_spot, y=pnl_current, mode="lines", name="Profit & Loss (actuel)", line=dict(color=current_color, dash="dash")))
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Payoff de la stratégie d'option FX (Short)",
            xaxis_title=f"Taux de change spot (MAD/{params['currency']})",
            yaxis_title=f"Profit / Loss (MAD)",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,  # Place la légende nettement sous le graphique
                xanchor="center",
                x=0.5,
                font=dict(size=20)
            ),
            margin={"l":40,"r":40,"t":90,"b":120},
            height=570
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🔎 <span style='color:#00FFFF'>Résultats Clés</span>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="center-table-container">
        <table class="result-table">
        <tr><td class="result-spot">Spot simulé</td><td>{spot_sim:.4f} {params['currency']}</td></tr>
        <tr><td class="result-strike">Strike</td><td>{strike:.4f} {params['currency']}</td></tr>
        <tr><td class="result-premium">Prime reçue</td><td>{premium:.4f} MAD</td></tr>
        <tr><td class="result-qty">Quantité</td><td>{qty:.0f}</td></tr>
        <tr><td class="result-mat">Maturité simulée</td><td>{mat_sim} jours</td></tr>
        <tr><td class="result-vol">Volatilité simulée</td><td>{sigma_sim:.2f} %</td></tr>
        <tr><td class="result-price">Prix option simulé</td><td>{option_val_sim:.4f} MAD</td></tr>
        <tr><td class="result-pnl">P&L dynamique</td><td>{profit:.2f} MAD</td></tr>
        <tr><td class="result-payoff">Payoff à maturité (au spot simulé)</td><td>{min(premium*qty, premium*qty - qty*direction*(spot_sim-strike)):.2f} MAD</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # =============== LONG ===============
    elif st.session_state["fx_strategy"] == "long":
        st.markdown("""
        <div class="neon-box-fixed">
        <b>Principe :</b><br>
        Acheter un <b>call</b> ou un <b>put</b> (“long call/put”) consiste à acquérir le droit (mais pas l’obligation) d’acheter ou vendre une devise à un prix fixé à l’avance.<br>
        La perte maximale est limitée à la prime payée, tandis que le gain potentiel peut être élevé si l’option termine largement dans la monnaie.<br>
        <b>Cette stratégie offre donc un profil de risque limité et un potentiel de gain asymétrique.</b>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="neon-box-fixed"> Choisissez <b>Long Call</b> ou <b>Long Put</b></div>
        <div class="neon-box-fixed"> Paramétrez la prime, le strike, la quantité</div>
        <div class="neon-box-fixed"> Visualisez le payoff, le P&L dynamique et les chiffres clés ci-dessous</div>
        """, unsafe_allow_html=True)
        option_type = st.selectbox("Stratégie", ["Call", "Put"], format_func=lambda x: f"Achat {x} (Long {x})", key="long_strat_type")
        premium = st.number_input("Prime payée (MAD)", min_value=0.01, value=1.0, step=0.01, key="long_premium")
        strike = st.number_input("Strike (MAD/Devise)", min_value=0.0001, value=params["spot"], step=0.0001, format="%.4f", key="long_strike")
        qty = st.number_input("Notionnel (en devise)", min_value=1.0, value=100000.0, step=1.0, format="%.0f", key="long_qty")

        st.markdown("#### Simulation dynamique")
        col_spot, col_mat, col_sigma = st.columns(3)
        with col_spot:
            spot_sim = st.slider("Spot FX simulé", min_value=0.0001, max_value=float(params["spot"]*4), value=float(params["spot"]), step=0.0001, format="%.4f", key="long_spot_sim")
        with col_mat:
            mat_sim = st.slider("Maturité simulée (jours)", min_value=1, max_value=int(params["maturity_days"]), value=int(params["maturity_days"]), step=1, key="long_mat_sim")
        with col_sigma:
            sigma_sim = st.slider("Volatilité simulée (%)", min_value=0.1, max_value=150.0, value=params["sigma"], step=0.1, key="long_sigma_sim")
        sigma_sim_ = sigma_sim / 100
        t_sim_ = mat_sim / 365

        if params["spot"] >= 50:
            stepSize = 0.5
        elif params["spot"] >= 10:
            stepSize = 0.1
        else:
            stepSize = 0.01
        x_spot = np.arange(max(0.0001, params["spot"] * 0.2), params["spot"] * 2 + stepSize, stepSize)
        direction = 1 if option_type == "Call" else -1
        payoff_maturity = [max(-premium * qty, -premium * qty + qty * direction * (xi - strike)) for xi in x_spot]
        option_current_prices = [garman_kohlhagen(xi, strike, t_sim_, params["r_d_"], params["r_f_"], sigma_sim_, option_type.lower()) for xi in x_spot]
        pnl_current = [(opc - premium) * qty for opc in option_current_prices]
        option_val_sim = garman_kohlhagen(spot_sim, strike, t_sim_, params["r_d_"], params["r_f_"], sigma_sim_, option_type.lower())
        profit = (option_val_sim - premium) * qty
        current_color = "green" if profit > 0.01 else "red" if profit < -0.01 else "black"

        st.markdown('<div class="fx-plot-neon">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_spot, y=payoff_maturity, mode="lines", name="Profit & Loss à maturité", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=x_spot, y=pnl_current, mode="lines", name="Profit & Loss (actuel)", line=dict(color=current_color, dash="dash")))
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Payoff de la stratégie d'option FX (Long)",
            xaxis_title=f"Taux de change spot (MAD/{params['currency']})",
            yaxis_title=f"Profit / Loss (MAD)",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=20)
            ),
            margin={"l":40,"r":40,"t":90,"b":120},
            height=570
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🔎 <span style='color:#00FFFF'>Résultats Clés</span>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="center-table-container">
        <table class="result-table">
        <tr><td class="result-spot">Spot simulé</td><td>{spot_sim:.4f} {params['currency']}</td></tr>
        <tr><td class="result-strike">Strike</td><td>{strike:.4f} {params['currency']}</td></tr>
        <tr><td class="result-premium">Prime payée</td><td>{premium:.4f} MAD</td></tr>
        <tr><td class="result-qty">Quantité</td><td>{qty:.0f}</td></tr>
        <tr><td class="result-mat">Maturité simulée</td><td>{mat_sim} jours</td></tr>
        <tr><td class="result-vol">Volatilité simulée</td><td>{sigma_sim:.2f} %</td></tr>
        <tr><td class="result-price">Prix option simulé</td><td>{option_val_sim:.4f} MAD</td></tr>
        <tr><td class="result-pnl">P&L dynamique</td><td>{profit:.2f} MAD</td></tr>
        <tr><td class="result-payoff">Payoff à maturité (au spot simulé)</td><td>{max(-premium*qty, -premium*qty + qty*direction*(spot_sim-strike)):.2f} MAD</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # =============== SPREAD ===============
    elif st.session_state["fx_strategy"] == "spread":
        st.markdown("""
        <div class="neon-box-fixed">
        <b>Principe :</b><br>
        Une <b>stratégie de spread</b> consiste à acheter et vendre la même quantité d'options (call ou put) à des strikes différents.<br>
        Le profil de risque est borné : gain et perte sont limités, ce qui réduit le coût d'entrée par rapport à un achat simple.<br>
        Le spread peut être haussier ou baissier selon la combinaison choisie.<br>
        <b>Cette approche permet de profiter d'un mouvement anticipé du marché avec une exposition maîtrisée.</b>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="neon-box-fixed"> Choisissez <b>Call Spread</b> ou <b>Put Spread</b></div>
        <div class="neon-box-fixed"> Paramétrez les deux jambes (primes, strikes, quantités, volatilités)</div>
        <div class="neon-box-fixed"> Visualisez le payoff, le P&L dynamique et les chiffres clés ci-dessous</div>
        """, unsafe_allow_html=True)
        option_type = st.selectbox("Stratégie", ["Call", "Put"], format_func=lambda x: f"{x} Spread", key="spread_strat_type")
        st.markdown("#### Jambe 1 (Long)")
        c_leg1 = st.columns(3)
        with c_leg1[0]:
            premium_1 = st.number_input("Prime 1 payée (MAD)", min_value=0.01, value=1.0, step=0.01, key="spread_premium1")
        with c_leg1[1]:
            strike_1 = st.number_input("Strike 1", min_value=0.0001, value=params["spot"], step=0.0001, format="%.4f", key="spread_strike1")
        with c_leg1[2]:
            qty_1 = st.number_input("Quantité 1 (notionnel)", min_value=1.0, value=100000.0, step=1.0, format="%.0f", key="spread_qty1")
        st.markdown("#### Jambe 2 (Short)")
        c_leg2 = st.columns(3)
        with c_leg2[0]:
            premium_2 = st.number_input("Prime 2 reçue (MAD)", min_value=0.01, value=0.8, step=0.01, key="spread_premium2")
        with c_leg2[1]:
            strike_2 = st.number_input("Strike 2", min_value=0.0001, value=params["spot"]*1.05, step=0.0001, format="%.4f", key="spread_strike2")
        with c_leg2[2]:
            qty_2 = st.number_input("Quantité 2 (notionnel)", min_value=1.0, value=100000.0, step=1.0, format="%.0f", key="spread_qty2")
        sigma1 = st.slider("Volatilité implicite Option 1 (%)", min_value=0.1, max_value=150.0, value=params["sigma"], step=0.1, key="spread_sigma1")
        sigma2 = st.slider("Volatilité implicite Option 2 (%)", min_value=0.1, max_value=150.0, value=params["sigma"], step=0.1, key="spread_sigma2")
        sigma1_ = sigma1 / 100
        sigma2_ = sigma2 / 100

        st.markdown("#### Simulation dynamique")
        col_spot, col_mat, col_sigma1, col_sigma2 = st.columns(4)
        with col_spot:
            spot_sim = st.slider("Spot FX simulé", min_value=0.0001, max_value=float(params["spot"]*4), value=float(params["spot"]), step=0.0001, format="%.4f", key="spread_spot_sim")
        with col_mat:
            mat_sim = st.slider("Maturité simulée (jours)", min_value=1, max_value=int(params["maturity_days"]), value=int(params["maturity_days"]), step=1, key="spread_mat_sim")
        with col_sigma1:
            sigma1_sim = st.slider("Volatilité simulée Option 1 (%)", min_value=0.1, max_value=150.0, value=sigma1, step=0.1, key="spread_sigma1_sim")
        with col_sigma2:
            sigma2_sim = st.slider("Volatilité simulée Option 2 (%)", min_value=0.1, max_value=150.0, value=sigma2, step=0.1, key="spread_sigma2_sim")
        sigma1_sim_ = sigma1_sim / 100
        sigma2_sim_ = sigma2_sim / 100
        t_sim_ = mat_sim / 365

        if params["spot"] >= 50:
            stepSize = 0.5
        elif params["spot"] >= 10:
            stepSize = 0.1
        else:
            stepSize = 0.01
        x_spot = np.arange(max(0.0001, params["spot"] * 0.2), params["spot"] * 2 + stepSize, stepSize)
        direction = 1 if option_type == "Call" else -1
        payoff_maturity = [
            qty_1 * (-premium_1 + max(0, direction * (xi-strike_1))) - qty_2 * (-premium_2 + max(0, direction * (xi-strike_2)))
            for xi in x_spot
        ]
        option1_prices = [garman_kohlhagen(xi, strike_1, t_sim_, params["r_d_"], params["r_f_"], sigma1_sim_, option_type.lower()) for xi in x_spot]
        option2_prices = [garman_kohlhagen(xi, strike_2, t_sim_, params["r_d_"], params["r_f_"], sigma2_sim_, option_type.lower()) for xi in x_spot]
        pnl_current = [
            (option1_prices[i] - premium_1) * qty_1 - (option2_prices[i] - premium_2) * qty_2
            for i in range(len(x_spot))
        ]
        option_val1_sim = garman_kohlhagen(spot_sim, strike_1, t_sim_, params["r_d_"], params["r_f_"], sigma1_sim_, option_type.lower())
        option_val2_sim = garman_kohlhagen(spot_sim, strike_2, t_sim_, params["r_d_"], params["r_f_"], sigma2_sim_, option_type.lower())
        profit = (option_val1_sim - premium_1) * qty_1 - (option_val2_sim - premium_2) * qty_2
        current_color = "green" if profit >= 0.01 else "red" if profit <= -0.01 else "black"

        st.markdown('<div class="fx-plot-neon">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_spot, y=payoff_maturity, mode="lines", name="Profit & Loss à maturité", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=x_spot, y=pnl_current, mode="lines", name="Profit & Loss (actuel)", line=dict(color=current_color, dash="dash")))
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Payoff de la stratégie d'option FX (Spread)",
            xaxis_title=f"Taux de change spot (MAD/{params['currency']})",
            yaxis_title=f"Profit / Loss (MAD)",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=20)
            ),
            margin={"l":40,"r":40,"t":90,"b":120},
            height=570
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🔎 <span style='color:#00FFFF'>Résultats Clés</span>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="center-table-container">
        <table class="result-table">
        <tr><td class="result-spot">Spot simulé</td><td>{spot_sim:.4f} {params['currency']}</td></tr>
        <tr><td class="result-strike">Strike 1</td><td>{strike_1:.4f} {params['currency']}</td></tr>
        <tr><td class="result-strike">Strike 2</td><td>{strike_2:.4f} {params['currency']}</td></tr>
        <tr><td class="result-premium">Prime 1 payée</td><td>{premium_1:.4f} MAD</td></tr>
        <tr><td class="result-premium">Prime 2 reçue</td><td>{premium_2:.4f} MAD</td></tr>
        <tr><td class="result-qty">Quantité 1</td><td>{qty_1:.0f}</td></tr>
        <tr><td class="result-qty">Quantité 2</td><td>{qty_2:.0f}</td></tr>
        <tr><td class="result-mat">Maturité simulée</td><td>{mat_sim} jours</td></tr>
        <tr><td class="result-vol">Volatilité simulée Option 1</td><td>{sigma1_sim:.2f} %</td></tr>
        <tr><td class="result-vol">Volatilité simulée Option 2</td><td>{sigma2_sim:.2f} %</td></tr>
        <tr><td class="result-price">Prix option 1 simulé</td><td>{option_val1_sim:.4f} MAD</td></tr>
        <tr><td class="result-price">Prix option 2 simulé</td><td>{option_val2_sim:.4f} MAD</td></tr>
        <tr><td class="result-pnl">P&L dynamique</td><td>{profit:.2f} MAD</td></tr>
        <tr><td class="result-payoff">Payoff à maturité (spot simulé)</td>
            <td>{qty_1 * (-premium_1 + max(0, direction * (spot_sim-strike_1))) - qty_2 * (-premium_2 + max(0, direction * (spot_sim-strike_2))):.2f} MAD</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # =============== STRADDLE ===============
    elif st.session_state["fx_strategy"] == "straddle":
        st.markdown("""
        <div class="neon-box-fixed">
        <b>Principe :</b><br>
        Un <b>straddle</b> consiste à acheter (ou vendre) la même quantité de calls et de puts au même strike, généralement at-the-money.<br>
        Le <b>long straddle</b> est intéressant lorsqu’on anticipe un fort mouvement (hausse ou baisse) sans savoir la direction.<br>
        Le <b>short straddle</b> (vente des deux jambes) parie sur une faible volatilité : le gain est limité, le risque potentiellement élevé.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="neon-box-fixed">Choisissez <b>Long Straddle</b> ou <b>Short Straddle</b></div>
        <div class="neon-box-fixed">Paramétrez les deux jambes (primes, strikes, quantités, volatilités)</div>
        <div class="neon-box-fixed">Visualisez le payoff, le P&L dynamique et les chiffres clés ci-dessous</div>
        """, unsafe_allow_html=True)
        strat_type = st.selectbox("Type de stratégie", ["LongStraddle", "ShortStraddle"], format_func=lambda x: "Long Straddle" if x=="LongStraddle" else "Short Straddle", key="straddle_strat_type")
        st.markdown("#### Jambe 1 (Call)")
        c_leg1 = st.columns(3)
        with c_leg1[0]:
            premium_call = st.number_input("Prime Call (MAD)", min_value=0.01, value=1.0, step=0.01, key="straddle_premium_call")
        with c_leg1[1]:
            strike_call = st.number_input("Strike Call", min_value=0.0001, value=params["spot"], step=0.0001, format="%.4f", key="straddle_strike_call")
        with c_leg1[2]:
            qty_call = st.number_input("Quantité Call (notionnel)", min_value=1.0, value=100000.0, step=1.0, format="%.0f", key="straddle_qty_call")
        st.markdown("#### Jambe 2 (Put)")
        c_leg2 = st.columns(3)
        with c_leg2[0]:
            premium_put = st.number_input("Prime Put (MAD)", min_value=0.01, value=1.0, step=0.01, key="straddle_premium_put")
        with c_leg2[1]:
            strike_put = st.number_input("Strike Put", min_value=0.0001, value=params["spot"], step=0.0001, format="%.4f", key="straddle_strike_put")
        with c_leg2[2]:
            qty_put = st.number_input("Quantité Put (notionnel)", min_value=1.0, value=100000.0, step=1.0, format="%.0f", key="straddle_qty_put")
        sigma_call = st.slider("Volatilité implicite Option Call (%)", min_value=0.1, max_value=150.0, value=params["sigma"], step=0.1, key="straddle_sigma_call")
        sigma_put = st.slider("Volatilité implicite Option Put (%)", min_value=0.1, max_value=150.0, value=params["sigma"], step=0.1, key="straddle_sigma_put")
        sigma_call_ = sigma_call / 100
        sigma_put_ = sigma_put / 100

        st.markdown("#### Simulation dynamique")
        col_spot, col_mat, col_sigma1, col_sigma2 = st.columns(4)
        with col_spot:
            spot_sim = st.slider("Spot FX simulé", min_value=0.0001, max_value=float(params["spot"]*4), value=float(params["spot"]), step=0.0001, format="%.4f", key="straddle_spot_sim")
        with col_mat:
            mat_sim = st.slider("Maturité simulée (jours)", min_value=1, max_value=int(params["maturity_days"]), value=int(params["maturity_days"]), step=1, key="straddle_mat_sim")
        with col_sigma1:
            sigma_call_sim = st.slider("Volatilité simulée Option Call (%)", min_value=0.1, max_value=150.0, value=sigma_call, step=0.1, key="straddle_sigma_call_sim")
        with col_sigma2:
            sigma_put_sim = st.slider("Volatilité simulée Option Put (%)", min_value=0.1, max_value=150.0, value=sigma_put, step=0.1, key="straddle_sigma_put_sim")
        sigma_call_sim_ = sigma_call_sim / 100
        sigma_put_sim_ = sigma_put_sim / 100
        t_sim_ = mat_sim / 365

        if params["spot"] >= 50:
            stepSize = 0.5
        elif params["spot"] >= 10:
            stepSize = 0.1
        else:
            stepSize = 0.01
        x_spot = np.arange(max(0.0001, params["spot"] * 0.2), params["spot"] * 2 + stepSize, stepSize)
        direction = 1 if strat_type == "LongStraddle" else -1
        payoff_maturity = [
            direction * (qty_call * (-premium_call + max(0, xi-strike_call)) + qty_put * (-premium_put + max(0, strike_put-xi)))
            for xi in x_spot
        ]
        call_prices = [garman_kohlhagen(xi, strike_call, t_sim_, params["r_d_"], params["r_f_"], sigma_call_sim_, "call") for xi in x_spot]
        put_prices = [garman_kohlhagen(xi, strike_put, t_sim_, params["r_d_"], params["r_f_"], sigma_put_sim_, "put") for xi in x_spot]
        pnl_current = [
            direction * ((call_prices[i] - premium_call) * qty_call + (put_prices[i] - premium_put) * qty_put)
            for i in range(len(x_spot))
        ]
        option_val_call_sim = garman_kohlhagen(spot_sim, strike_call, t_sim_, params["r_d_"], params["r_f_"], sigma_call_sim_, "call")
        option_val_put_sim = garman_kohlhagen(spot_sim, strike_put, t_sim_, params["r_d_"], params["r_f_"], sigma_put_sim_, "put")
        profit = direction * ((option_val_call_sim - premium_call) * qty_call + (option_val_put_sim - premium_put) * qty_put)
        current_color = "green" if profit >= 0.01 else "red" if profit <= -0.01 else "black"

        st.markdown('<div class="fx-plot-neon">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_spot, y=payoff_maturity, mode="lines", name="Profit & Loss à maturité", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=x_spot, y=pnl_current, mode="lines", name="Profit & Loss (actuel)", line=dict(color=current_color, dash="dash")))
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Payoff de la stratégie d'option FX (Straddle)",
            xaxis_title=f"Taux de change spot (MAD/{params['currency']})",
            yaxis_title=f"Profit / Loss (MAD)",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=20)
            ),
            margin={"l":40,"r":40,"t":90,"b":120},
            height=570
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🔎 <span style='color:#00FFFF'>Résultats Clés</span>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="center-table-container">
        <table class="result-table">
        <tr><td class="result-spot">Spot simulé</td><td>{spot_sim:.4f} {params['currency']}</td></tr>
        <tr><td class="result-strike">Strike Call</td><td>{strike_call:.4f} {params['currency']}</td></tr>
        <tr><td class="result-strike">Strike Put</td><td>{strike_put:.4f} {params['currency']}</td></tr>
        <tr><td class="result-premium">Prime Call</td><td>{premium_call:.4f} MAD</td></tr>
        <tr><td class="result-premium">Prime Put</td><td>{premium_put:.4f} MAD</td></tr>
        <tr><td class="result-qty">Quantité Call</td><td>{qty_call:.0f}</td></tr>
        <tr><td class="result-qty">Quantité Put</td><td>{qty_put:.0f}</td></tr>
        <tr><td class="result-mat">Maturité simulée</td><td>{mat_sim} jours</td></tr>
        <tr><td class="result-vol">Volatilité simulée Call</td><td>{sigma_call_sim:.2f} %</td></tr>
        <tr><td class="result-vol">Volatilité simulée Put</td><td>{sigma_put_sim:.2f} %</td></tr>
        <tr><td class="result-price">Prix Call simulé</td><td>{option_val_call_sim:.4f} MAD</td></tr>
        <tr><td class="result-price">Prix Put simulé</td><td>{option_val_put_sim:.4f} MAD</td></tr>
        <tr><td class="result-pnl">P&L dynamique</td><td>{profit:.2f} MAD</td></tr>
        <tr><td class="result-payoff">Payoff à maturité (spot simulé)</td>
            <td>{direction * (qty_call * (-premium_call + max(0, spot_sim-strike_call)) + qty_put * (-premium_put + max(0, strike_put-spot_sim))):.2f} MAD</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # =============== STRANGLE ===============
    elif st.session_state["fx_strategy"] == "strangle":
        st.markdown("""
        <div class="neon-box-fixed">
        <b>Principe :</b><br>
        La <b>stratégie strangle</b> consiste à acheter (ou vendre) la même quantité de calls et de puts, mais à des strikes différents, généralement tous deux hors de la monnaie.<br>
        Elle permet de profiter d'un mouvement important (hausse ou baisse) de l'actif sous-jacent, sans préjuger de la direction.<br>
        Le <b>long strangle</b> est moins coûteux qu’un straddle mais nécessite un mouvement plus ample ; à l'inverse, le <b>short strangle</b> parie sur une faible volatilité, avec un risque important.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="neon-box-fixed">Choisissez <b>Long Strangle</b> ou <b>Short Strangle</b></div>
        <div class="neon-box-fixed">Paramétrez les deux jambes (primes, strikes, quantités, volatilités)</div>
        <div class="neon-box-fixed">Visualisez le payoff, le P&L dynamique et les chiffres clés ci-dessous</div>
        """, unsafe_allow_html=True)
        strat_type = st.selectbox("Type de stratégie", ["LongStrangle", "ShortStrangle"], format_func=lambda x: "Long Strangle" if x=="LongStrangle" else "Short Strangle", key="strangle_strat_type")
        st.markdown("#### Jambe 1 (Call)")
        c_leg1 = st.columns(3)
        with c_leg1[0]:
            premium_call = st.number_input("Prime Call (MAD)", min_value=0.01, value=1.0, step=0.01, key="strangle_premium_call")
        with c_leg1[1]:
            strike_call = st.number_input("Strike Call", min_value=0.0001, value=params["spot"]*1.09, step=0.0001, format="%.4f", key="strangle_strike_call")
        with c_leg1[2]:
            qty_call = st.number_input("Quantité Call (notionnel)", min_value=1.0, value=100000.0, step=1.0, format="%.0f", key="strangle_qty_call")
        st.markdown("#### Jambe 2 (Put)")
        c_leg2 = st.columns(3)
        with c_leg2[0]:
            premium_put = st.number_input("Prime Put (MAD)", min_value=0.01, value=1.0, step=0.01, key="strangle_premium_put")
        with c_leg2[1]:
            strike_put = st.number_input("Strike Put", min_value=0.0001, value=params["spot"]*0.91, step=0.0001, format="%.4f", key="strangle_strike_put")
        with c_leg2[2]:
            qty_put = st.number_input("Quantité Put (notionnel)", min_value=1.0, value=100000.0, step=1.0, format="%.0f", key="strangle_qty_put")
        sigma_call = st.slider("Volatilité implicite Option Call (%)", min_value=0.1, max_value=150.0, value=params["sigma"], step=0.1, key="strangle_sigma_call")
        sigma_put = st.slider("Volatilité implicite Option Put (%)", min_value=0.1, max_value=150.0, value=params["sigma"], step=0.1, key="strangle_sigma_put")
        sigma_call_ = sigma_call / 100
        sigma_put_ = sigma_put / 100

        st.markdown("#### Simulation dynamique")
        col_spot, col_mat, col_sigma1, col_sigma2 = st.columns(4)
        with col_spot:
            spot_sim = st.slider("Spot FX simulé", min_value=0.0001, max_value=float(params["spot"]*4), value=float(params["spot"]), step=0.0001, format="%.4f", key="strangle_spot_sim")
        with col_mat:
            mat_sim = st.slider("Maturité simulée (jours)", min_value=1, max_value=int(params["maturity_days"]), value=int(params["maturity_days"]), step=1, key="strangle_mat_sim")
        with col_sigma1:
            sigma_call_sim = st.slider("Volatilité simulée Call (%)", min_value=0.1, max_value=150.0, value=sigma_call, step=0.1, key="strangle_sigma_call_sim")
        with col_sigma2:
            sigma_put_sim = st.slider("Volatilité simulée Put (%)", min_value=0.1, max_value=150.0, value=sigma_put, step=0.1, key="strangle_sigma_put_sim")
        sigma_call_sim_ = sigma_call_sim / 100
        sigma_put_sim_ = sigma_put_sim / 100
        t_sim_ = mat_sim / 365

        if params["spot"] >= 50:
            stepSize = 0.5
        elif params["spot"] >= 10:
            stepSize = 0.1
        else:
            stepSize = 0.01
        x_spot = np.arange(max(0.0001, params["spot"] * 0.2), params["spot"] * 2 + stepSize, stepSize)
        direction = 1 if strat_type == "LongStrangle" else -1
        payoff_maturity = [
            direction * (qty_call * (-premium_call + max(0, xi-strike_call)) + qty_put * (-premium_put + max(0, strike_put-xi)))
            for xi in x_spot
        ]
        call_prices = [garman_kohlhagen(xi, strike_call, t_sim_, params["r_d_"], params["r_f_"], sigma_call_sim_, "call") for xi in x_spot]
        put_prices = [garman_kohlhagen(xi, strike_put, t_sim_, params["r_d_"], params["r_f_"], sigma_put_sim_, "put") for xi in x_spot]
        pnl_current = [
            direction * ((call_prices[i] - premium_call) * qty_call + (put_prices[i] - premium_put) * qty_put)
            for i in range(len(x_spot))
        ]
        option_val_call_sim = garman_kohlhagen(spot_sim, strike_call, t_sim_, params["r_d_"], params["r_f_"], sigma_call_sim_, "call")
        option_val_put_sim = garman_kohlhagen(spot_sim, strike_put, t_sim_, params["r_d_"], params["r_f_"], sigma_put_sim_, "put")
        profit = direction * ((option_val_call_sim - premium_call) * qty_call + (option_val_put_sim - premium_put) * qty_put)
        current_color = "green" if profit >= 0.01 else "red" if profit <= -0.01 else "black"

        st.markdown('<div class="fx-plot-neon">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_spot, y=payoff_maturity, mode="lines", name="Profit & Loss à maturité", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=x_spot, y=pnl_current, mode="lines", name="Profit & Loss (actuel)", line=dict(color=current_color, dash="dash")))
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Payoff de la stratégie d'option FX (Strangle)",
            xaxis_title=f"Taux de change spot (MAD/{params['currency']})",
            yaxis_title=f"Profit / Loss (MAD)",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=20)
            ),
            margin={"l":40,"r":40,"t":90,"b":120},
            height=570
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🔎 <span style='color:#00FFFF'>Résultats Clés</span>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="center-table-container">
        <table class="result-table">
        <tr><td class="result-spot">Spot simulé</td><td>{spot_sim:.4f} {params['currency']}</td></tr>
        <tr><td class="result-strike">Strike Call</td><td>{strike_call:.4f} {params['currency']}</td></tr>
        <tr><td class="result-strike">Strike Put</td><td>{strike_put:.4f} {params['currency']}</td></tr>
        <tr><td class="result-premium">Prime Call</td><td>{premium_call:.4f} MAD</td></tr>
        <tr><td class="result-premium">Prime Put</td><td>{premium_put:.4f} MAD</td></tr>
        <tr><td class="result-qty">Quantité Call</td><td>{qty_call:.0f}</td></tr>
        <tr><td class="result-qty">Quantité Put</td><td>{qty_put:.0f}</td></tr>
        <tr><td class="result-mat">Maturité simulée</td><td>{mat_sim} jours</td></tr>
        <tr><td class="result-vol">Volatilité simulée Call</td><td>{sigma_call_sim:.2f} %</td></tr>
        <tr><td class="result-vol">Volatilité simulée Put</td><td>{sigma_put_sim:.2f} %</td></tr>
        <tr><td class="result-price">Prix Call simulé</td><td>{option_val_call_sim:.4f} MAD</td></tr>
        <tr><td class="result-price">Prix Put simulé</td><td>{option_val_put_sim:.4f} MAD</td></tr>
        <tr><td class="result-pnl">P&L dynamique</td><td>{profit:.2f} MAD</td></tr>
        <tr><td class="result-payoff">Payoff à maturité (spot simulé)</td>
            <td>{direction * (qty_call * (-premium_call + max(0, spot_sim-strike_call)) + qty_put * (-premium_put + max(0, strike_put-spot_sim))):.2f} MAD</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)         

with tab8:
    import numpy as np
    import streamlit as st
    import plotly.graph_objs as go
    from scipy.stats import norm

    # ------ CSS ---------
    st.markdown(
        """
        <style>
        body { background: #181b2a; }
        .block-container { padding-left: 1rem; padding-right: 1rem; max-width: 100vw; }
        .intro-box {
            background: rgba(15, 15, 30, 0.92);
            border: 2px solid rgba(0, 255, 255, 0.7);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 0 18px 2px #00ffff55, 0 0 45px 10px #00ffff22 inset;
            color: white;
            font-size: 16px;
            line-height: 1.6;
            backdrop-filter: blur(6px);
        }
        .neon-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0 20px 0;
            background: rgba(30,40,60,0.95);
            color: #fff;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 0 10px #00ffff33;
        }
        .neon-table th, .neon-table td {
            border: 1px solid #00d1d1;
            padding: 10px 14px;
            text-align: left;
        }
        .neon-table th {
            background: #0c222a;
            color: #00ffd7;
        }
        .neon-table tr:nth-child(even) {
            background-color: #1b2640;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("FX BOOK – Tunnel Symétrique & Asymétrique")

    # --------- Ce que propose ce module (explication en haut) ----------
    st.markdown("""
    <div style="background: rgba(15, 23, 45, 0.95); border: 2.5px solid #00ffd7cc; border-radius: 18px;
         padding: 20px 24px; margin-bottom: 24px; color: #e8fcff; font-size:17px; font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;">
        <b>Ce module vous permet de :</b>
        <ul>
            <li>Simuler le pricing d’un tunnel de change (symétrique ou asymétrique)</li>
            <li>Comparer les primes et les scénarios à maturité</li>
            <li>Visualiser le payoff en fonction du spot à l’échéance</li>
            <li>Comprendre le mécanisme de couverture via un affichage pédagogique</li>
        </ul>
        <i>Idéal pour explorer l'intérêt des stratégies de tunnels en gestion du risque de change.</i>
    </div>
    """, unsafe_allow_html=True)

    # --------- Bouton description affichage/masquage ---------
    if "show_desc_tunnel" not in st.session_state:
        st.session_state["show_desc_tunnel"] = False
    if st.button("Afficher / masquer la description des tunnels"):
        st.session_state["show_desc_tunnel"] = not st.session_state["show_desc_tunnel"]

    if st.session_state["show_desc_tunnel"]:
        st.markdown("""
        <div class="intro-box" style="margin-bottom: 16px;">
            <b>Tunnel Symétrique</b>
            <table class="neon-table">
            <tr><th>Plancher (Put vendu)</th><td>K1</td></tr>
            <tr><th>Plafond (Call acheté)</th><td>K2</td></tr>
            <tr><th>Nominal</th><td>N</td></tr>
            <tr><th>Prime</th><td>Faible voire nulle</td></tr>
            <tr><th>Scénarios</th>
            <td>
            S ≥ K2 : Achat à K2<br>
            S &lt; K1 : Achat à K1<br>
            K1 &lt; S &lt; K2 : Achat au Spot
            </td></tr>
            </table>
            <ul>
            <li>Garantie d’un cours maximum d’achat (K2)</li>
            <li>Possibilité de profiter de la baisse</li>
            <li>Profit limité en cas de forte baisse</li>
            </ul>
            Exemple :<br>
            - Spot 10.95 ⇒ Achat à 10.90<br>
            - Spot 10.70 ⇒ Achat à 10.80<br>
            - Spot 10.85 ⇒ Achat à 10.85
        </div>
        <div class="intro-box">
            <b>Tunnel Asymétrique</b>
            <table class="neon-table">
            <tr><th>Plancher (Put vendu)</th><td>K1, nominal m×N</td></tr>
            <tr><th>Plafond (Call acheté)</th><td>K2, nominal 1×N</td></tr>
            <tr><th>Prime</th><td>Faible voire nulle</td></tr>
            <tr><th>Scénarios</th>
            <td>
            S &gt; K2 : Achat 1×N à K2<br>
            S &lt; K1 : Achat m×N à K1<br>
            K1 &lt; S &lt; K2 : Achat au Spot (1×N)
            </td></tr>
            </table>
            <ul>
            <li>Couloir plus large</li>
            <li>Pas de prime à payer</li>
            <li>Incertitude sur le nominal à l’échéance</li>
            </ul>
            Exemple :<br>
            - Spot 10.95 ⇒ Achat 1×N à 10.90<br>
            - Spot 10.70 ⇒ Achat m×N à 10.80<br>
            - Spot 10.85 ⇒ Achat à 10.85
        </div>
        """, unsafe_allow_html=True)
    
    # ----------- Inputs ----------------
    st.header("Paramètres du Pricing")
    colA, colB, colC = st.columns(3)
    with colA:
        tunnel_type = st.selectbox("Type de Tunnel", ["Symétrique", "Asymétrique"], key="tunnel_type_tab8")
        S = st.number_input("Taux Spot (S)", value=10.5)
    with colB:
        K1 = st.number_input("Strike Put K1 (plancher)", value=10.0)
        K2 = st.number_input("Strike Call K2 (plafond)", value=11.0)
    with colC:
        T = st.number_input("Maturité (en années)", value=0.5)
        r_d = st.number_input("Taux domestique r_d", value=0.03)
        r_f = st.number_input("Taux étranger r_f", value=0.01)
        sigma = st.number_input("Volatilité (ex: 0.15 pour 15%)", value=0.15)
        nominal = st.number_input("Nominal (en devise)", value=1000000)
        if tunnel_type == "Asymétrique":
            m = st.slider("Ratio m (Put vendus / Call achetés)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        else:
            m = 1.0

    # --------- Pricing Functions ---------
    def garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type):
        if T <= 0:
            raise ValueError("La maturité doit être positive")
        d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if option_type == 'call':
            return S * np.exp(-r_f*T) * norm.cdf(d1) - K * np.exp(-r_d*T) * norm.cdf(d2)
        elif option_type == 'put':
            return K * np.exp(-r_d*T) * norm.cdf(-d2) - S * np.exp(-r_f*T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")

    def pricer_tunnel_sym(S, K1, K2, T, r_d, r_f, sigma, nominal):
        call_price = garman_kohlhagen(S, K2, T, r_d, r_f, sigma, 'call')
        put_price = garman_kohlhagen(S, K1, T, r_d, r_f, sigma, 'put')
        net_premium = call_price - put_price
        return {
            "Call (K2)": call_price * nominal,
            "Put (K1)": put_price * nominal,
            "Prime nette": net_premium * nominal
        }

    def pricer_tunnel_asym(S, K1, K2, T, r_d, r_f, sigma, nominal, m):
        call_price = garman_kohlhagen(S, K2, T, r_d, r_f, sigma, 'call')
        put_price = garman_kohlhagen(S, K1, T, r_d, r_f, sigma, 'put')
        net_premium = call_price * nominal - put_price * nominal * m
        return {
            "Call (K2), 1N": call_price * nominal,
            "Put (K1), mN": put_price * nominal * m,
            "Prime nette": net_premium
        }

    # --------- Payoff Function (à maturité) ---------
    def payoff_tunnel_sym(spot, K1, K2, nominal):
        payoff = []
        for S_T in spot:
            if S_T >= K2:
                payoff.append(nominal * (K2 - S_T))
            elif S_T <= K1:
                payoff.append(nominal * (K1 - S_T))
            else:
                payoff.append(0)
        return payoff

    def payoff_tunnel_asym(spot, K1, K2, nominal, m):
        payoff = []
        for S_T in spot:
            if S_T >= K2:
                payoff.append(nominal * (K2 - S_T))
            elif S_T <= K1:
                payoff.append(m * nominal * (K1 - S_T))
            else:
                payoff.append(0)
        return payoff

    # --------- Scénarios Tableau ---------
    def scenarios_tableau(tunnel_type, K1, K2, nominal, m):
        if tunnel_type == "Symétrique":
            rows = [
                ("Spot < K1", f"Achat à K1", f"{nominal:,.0f}", f"{K1:,.4f}"),
                (f"K1 ≤ Spot ≤ K2", f"Achat au Spot", f"{nominal:,.0f}", "Spot"),
                (f"Spot > K2", f"Achat à K2", f"{nominal:,.0f}", f"{K2:,.4f}")
            ]
        else:
            rows = [
                ("Spot < K1", f"Achat à K1", f"{nominal*m:,.0f}", f"{K1:,.4f}"),
                (f"K1 ≤ Spot ≤ K2", f"Achat au Spot", f"{nominal:,.0f}", "Spot"),
                (f"Spot > K2", f"Achat à K2", f"{nominal:,.0f}", f"{K2:,.4f}")
            ]
        html = '<table class="neon-table"><tr><th>Scénario</th><th>Prix d\'achat effectif</th><th>Nominal traité</th><th>Cours d\'exécution</th></tr>'
        for row in rows:
            html += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        html += "</table>"
        return html

    # ----------- Compute Button -----------
    if st.button("💰 Afficher les Résultats"):
        st.header("Résumé du Pricing")
        if tunnel_type == "Symétrique":
            result = pricer_tunnel_sym(S, K1, K2, T, r_d, r_f, sigma, nominal)
        else:
            result = pricer_tunnel_asym(S, K1, K2, T, r_d, r_f, sigma, nominal, m)

        st.markdown("### 📋 Résultats")
        st.markdown(
            f"""
            <table class="neon-table">
                {''.join(f'<tr><th>{k}</th><td>{round(v,2):,.2f} MAD</td></tr>' for k,v in result.items())}
            </table>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 🧩 <span style='color:#00ffff'>Scénarios à maturité</span>", unsafe_allow_html=True)
        st.markdown(scenarios_tableau(tunnel_type, K1, K2, nominal, m), unsafe_allow_html=True)

        st.markdown("### 📊 Payoff à maturité")
        spot_sim = np.linspace(K1-1, K2+1, 120)
        if tunnel_type == "Symétrique":
            payoff = payoff_tunnel_sym(spot_sim, K1, K2, nominal)
        else:
            payoff = payoff_tunnel_asym(spot_sim, K1, K2, nominal, m)

        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(
            x=spot_sim, y=payoff, mode='lines+markers',
            name="Payoff à maturité", line=dict(color="#00ffe7")
        ))
        fig_payoff.add_hline(y=0, line_color="#00c2ff", line_dash="dot")
        fig_payoff.update_layout(
            title="Payoff tunnel à maturité selon le spot à l'échéance",
            xaxis_title="Spot à l'échéance",
            yaxis_title="Gain (MAD)",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5, font=dict(size=19)),
            margin={"l":40,"r":40,"t":80,"b":100},
            height=420,
            plot_bgcolor="#1b2640",
            paper_bgcolor="#181b2a"
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

       
