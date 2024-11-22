# ELIZA 

## Histoire

ELIZA est, en intelligence artificielle, un programme informatique écrit par Joseph Weizenbaum entre 1964 et 1966, qui simule un psychothérapeute rogérien en reformulant la plupart des affirmations du « patient » en questions, et en les lui posant.

*source: wikipedia* 

## Installer les dépendances python 


### Créer un env python 

```
python -m venv venv
```




docker compose from : https://github.com/valiantlynx/ollama-docker

docker exec -it ollama bash    

ollama help
ollama run llama3.2:1b


>>> /set parameter

>>>/show parameters 



Available Parameters:
  /set parameter seed <int>             Random number seed
  /set parameter num_predict <int>      Max number of tokens to predict
  /set parameter top_k <int>            Pick from top k num of tokens
  /set parameter top_p <float>          Pick token based on sum of probabilities
  /set parameter min_p <float>          Pick token based on top token probability * min_p
  /set parameter num_ctx <int>          Set the context size
  /set parameter temperature <float>    Set creativity level
  /set parameter repeat_penalty <float> How strongly to penalize repetitions
  /set parameter repeat_last_n <int>    Set how far back to look for repetitions
  /set parameter num_gpu <int>          The number of layers to send to the GPU
  /set parameter stop <string> <string> ...   Set the stop parameters