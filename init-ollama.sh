#!/bin/bash

# Lancer le service Ollama
ollama serve &

# Attendre que le service démarre
sleep 5

# Télécharger les modèles nécessaires
if ! ollama list | grep -q "llama3.2:1b"; then
    echo "Téléchargement du modèle llama3.2:1b..."
    ollama pull llama3.2:1b
fi

if ! ollama list | grep -q "nomic-embed-text"; then
    echo "Téléchargement du modèle nomic-embed-text..."
    ollama pull nomic-embed-text:latest
fi

# Optionnel : Lancer une vérification initiale des modèles
echo "Test llama3.2:1b" | ollama run llama3.2:1b


# Garder le processus en cours d'exécution
tail -f /dev/null