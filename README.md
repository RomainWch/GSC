# GSC
IA agent jouant à Pong et LunarLander (DQN)

## Résumé
Ce dépôt contient une implémentation DQN simple pour entraîner un agent sur des environnements OpenAI Gym (via `gymnasium`), notamment `LunarLander-v3`.

**Prérequis**
- Python : `python3` (recommandé >= 3.8)

## Installation (Debian/Ubuntu)

1. Créer et activer un environnement virtuel Python :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installer tous les paquets Python requis en une seule commande (le fichier `requirements.txt` contient la liste) :

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
# l'installation peut mettre un peu de temps
```

si l'installation de box2d échoue essayez :

```bash
sudo apt install -y swig python3-dev build-essential
pip install "gymnasium[box2d]"
```

puis relancer : 
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
# l'installation peut mettre un peu de temps
```

## Lancer l'entraînement
- Après activation du virtualenv et installation des dépendances :

- Pour LunarLander, lancez :
```bash
# depuis la racine du dépôt (où se trouve `run_agentLL.py`)
python3 run_agentLL.py
```

-Pour un jeu de Atari, lancez :
```bash
# depuis la racine du dépôt (où se trouve `run_agentALE.py`)
python3 run_agentALE.py 
```
avec :
--env ALE/Pong-v5 pour jouer à pong ou ALE/Breakout-v5 pour jouer au casse brique (par défaut : ALE/Pong-v5)
--episodes suivi du nombre d'épisodes d'apprentissage (par défaut : 1501)

Pour l'apprentissage d'un jeu Atari il n'y a pas de render lors de l'apprentissage pour des raisons pratiques.

## Jouer une stratégie
- Après activation du virtualenv et installation des dépendances, lancez :

```bash
# depuis la racine du dépôt (où se trouve `play_policy.py`)
python3 play_policy.py policies/policiesName.pth 
```

avec :
--env ALE/Pong-v5 pour pong, ALE/Breakout-v5 pour le casse brique ou LunarLander-v3 pour LunarLander (par défaut : ALE/Pong-v5)
--episodes suivi du nombre d'episodes à jouer (par défaut : 5)
--no-render pour ne pas afficher le jeu, ne rien mettre pour l'afficher

Exemples : 

- Pour LunarLander
```bash
# depuis la racine du dépôt (où se trouve `play_policy.py`)
python3 play_policy.py policies/lunar_test_numero_43_avg263_ep900.pth --env LunarLander-v3 --episodes 3 
```

- Pour Pong
```bash
# depuis la racine du dépôt (où se trouve `play_policy.py`)
python3 play_policy.py policies/Pong_test_numero_2_avg-9_ep1200.pth --env ALE/Pong-v5 --episodes 3
```