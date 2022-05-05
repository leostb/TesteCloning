import random

import matplotlib.pyplot as plt
from numpy import cos
from numpy import e
from numpy import exp
from numpy import pi
from numpy import sqrt

# TODO incluir bounds
# TODO usar menos precisão decimal?

estatistica = {}


def algoritmo_genetico(numero_epocas, func, probabilidade_cross, probabilidade_mutacao, tamanho_pop, lower, upper):
    populacao = inicializar_população(tamanho_pop, lower, upper)
    populacao = calcular_fitness(populacao, func)
    populacao = ordenar(populacao)
    plt.pie(populacao.values(), labels=populacao.keys())
    plt.title("População inicial")
    plt.show()
    for i in range(0, numero_epocas):
        print("**** Geracao " + str(i) + "*****")
        popcross = crossover(populacao, probabilidade_cross, func)  # Reprodução
        # mutacao(populacao, probabilidade_mutacao)  # TODO vai pegar os pais também, não deveria ter mutação nos pais
        # array_fitness = calcular_fitness(populacao, func)
        populacao.update(popcross)
        populacao = seleciona(populacao, tamanho_pop)  # Escolher um método tipo roleta
        populacao = calcular_fitness(populacao, func)
    return populacao


def inicializar_população(tamanho_pop, lower, upper):
    # Retorna um array de pares x,y
    # Poderia já calcular o fitness aqui, mas por questão de didatica vou calcular depoist
    pop = {random.uniform(lower, upper): 0 for i in range(tamanho_pop)}
    return pop


def calcular_fitness(populacao, func):
    for k, v in populacao.items():
        populacao.update({k: func(k)})
    # TODO testei e não precisava retornar, ele já alterou o objeto original
    return populacao


def crossover(populacao, pc, func):
    populacao2 = {}
    for filho, v in populacao.items():
        if random.random() < pc:
            novofilho = filho + roleta(populacao) / 2
            populacao2.update({novofilho: func(novofilho)})
        else:
            populacao2.update({filho: v})

    return populacao2


def mutacao(população, pm):
    # if (np.rand() > pm):
    #     for filho in população
    pass


def seleciona(populacao, tamanho_pop):
    nova_pop = {}
    while len(nova_pop) < tamanho_pop:
        nova_pop.update({roleta(populacao): -999})
    return nova_pop


def roleta(populacao):
    # TODO melhorar ficou muito tosco hausashu muita conversão de tipo para fazer isso
    # TODO talvez tenha um método que receba apenas o dicionario
    sorteado = random.choices(list(populacao.keys()), list(populacao.values())).pop()
    contabilizar_roleta(sorteado)
    # plt.pie(list(map(abs, populacao.keys())), list(map(abs, populacao.values())), normalize=True)
    # plt.title("Roleta")
    # plt.show()
    return sorteado


def ordenar(populacao):
    dicionario_ordenado = {}
    chaves_ordenadas = sorted(populacao.items(), key=lambda x: x[1], reverse=True)

    for w in chaves_ordenadas:
        dicionario_ordenado[w[0]] = w[1]

    return dicionario_ordenado


def contabilizar_roleta(chave):
    if chave not in estatistica:
        estatistica[chave] = 1
    else:
        estatistica[chave] += 1


def objective(x):
    return 1 / (x ** 2.0)


def ackley(x, y):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(0.5 * (cos(2 *
                                                                              pi * x) + cos(2 * pi * y))) + e + 20


if __name__ == '__main__':
    tamPop = 10
    pc = 0.9  # Probabilidade de Crossover
    pm = 0.1  # Probabilidade de mutação
    Ngera = 100  # Nro de gerações
    Nbits = 32  # Número de bits para cada variável
    Nvar = 2  # Nro de variáveis
    gera = 0  # Geração inicial
    lower = -5
    upper = 5
    # func =
    pop_final = algoritmo_genetico(Ngera, objective, pc, pm, tamPop, lower, upper)
    plt.pie(pop_final.values(), labels=pop_final.keys())
    plt.title("População final")
    plt.show()
    print(str(pop_final))
