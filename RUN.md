# Recommandations musicales — utilisation

1. Installer : `python -m pip install -r requirements.txt`.
2. Exécuter le notebook ou lancer `python music_rag.py "relaxing acoustic guitar"`.
3. Pour activer le RAG entièrement local : installer Ollama, démarrer son service, puis lancer `ollama pull qwen2.5:3b`.
4. Générer : `python music_rag.py "relaxing acoustic guitar" --generate`, ou activer `RUN_LOCAL_LLM=True` dans le notebook.
5. Tests : `python -m unittest -v test_music_rag.py`.

Le catalogue MusicCaps est inclus : 5 521 descriptions réelles, tags et liens vers des extraits YouTube. Aucun titre ni artiste n’est inventé. La recherche locale ne nécessite ni Pinecone ni clé OpenAI. Sans Ollama, l’affichage est une restitution des sources retrouvées, pas une génération LLM.

L’index TF-IDF est reconstruit en mémoire depuis le CSV à chaque lancement. Les requêtes françaises bénéficient d’un petit dictionnaire d’expansion ; il ne s’agit pas d’une traduction complète. Les filtres `required_terms` et `excluded_terms` recherchent des sous-chaînes dans les métadonnées. Pour exclure explicitement une caractéristique, utiliser ces filtres au lieu d’attendre une compréhension générale des négations.

La démonstration audio originale figure en annexe facultative. Elle a des dépendances séparées et nécessite une connexion à Pinecone. Elle n’a pas été exécutée pendant la validation de la solution locale.
