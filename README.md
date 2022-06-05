# Exercício Programa 1 ACH-2016  

Este código é referente ao  exercício de algoritmos genéticos da disciplina de Inteligência
artificial EACH-USP 2022

## Descrição

Exercita a otimização de 6 funções (gold, summs, dejong, ackley, bump, rastrigin)
através de algoritmo genético.

Operadors:
- Cross-over
- Mutação
- Seleção

## Rodando
Instale as dependências antes de executar o programa (main.py)

### Dependências

* Python 3
* Pip3 (gerenciador de pacotes)
* Numpy, matplotlib, ... (todas as bibliotecas estão no arquivo requirements.txt)

### Instalando dependências
Depois de instalado o pip3. Execute
```
pip install -r requirements.txt
```

### Executando o programa

```
python3 main.py
```

## Explorando os parâmetros

Os parâmetros estão hard coded no código, para experimentar mudanças 
basta alterá-los

## Functions.py

O arquivo functions.py o código das funçoes utilizadas. Mas se 
você executá-lo ele plota o gráfico e as curvas de nível.

```
python3 functions.py
```


## Relatórios

Cada execução gera um relatório na pasta `./relatorios`