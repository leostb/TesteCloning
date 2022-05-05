import random

import matplotlib.pyplot as plt
from numpy import cos
from numpy import e
from numpy import exp
from numpy import pi
from numpy import sqrt
import copy

# TODO incluir bounds
# TODO usar menos precisão decimal?

estatistica = {}


def algoritmo_genetico(numero_epocas, func, probabilidade_cross, probabilidade_mutacao, tamanho_pop, lower, upper):
    populacao = inicializar_população(tamanho_pop, lower, upper)
    populacao = calcular_fitness(populacao, func)
    populacao = ordenar(populacao)
    plt.pie([row[2] for row in populacao], labels=['{}, {}'.format(row[0], row[1]) for row in populacao])
    plt.title("População inicial")
    plt.show()
    for i in range(0, numero_epocas):
        print("**** Geracao " + str(i) + "*****")
        popcross = crossover(populacao, probabilidade_cross, func)  # Reprodução
        # mutacao(populacao, probabilidade_mutacao)  # TODO vai pegar os pais também, não deveria ter mutação nos pais
        # array_fitness = calcular_fitness(populacao, func)
        populacao += popcross
        populacao = seleciona(populacao, tamanho_pop)  # Escolher um método tipo roleta
        populacao = calcular_fitness(populacao, func)
    return populacao


def inicializar_população(tamanho_pop, lower, upper):
    # Retorna um array de pares x,y
    # Poderia já calcular o fitness aqui, mas por questão de didatica vou calcular depoist
    populacao=[]
    for i in range(tamanho_pop):
        populacao.append([random.uniform(lower, upper), random.uniform(lower, upper), 0])
    return populacao


def calcular_fitness(populacao, func):
    for item in populacao:
        item[2] = func(item[0], item[1]) # TODO Refatorar, ao invés de trabalhar com lista, poderia fazer uma classe
    # TODO testei e não precisava retornar, ele já alterou o objeto original
    return populacao


def roleta(populacao):
    # TODO melhorar ficou muito tosco hausashu muita conversão de tipo para fazer isso
    # TODO talvez tenha um método que receba apenas o dicionario
    sorteado = random.choices(populacao, [row[2] for row in populacao]).pop()
    #contabilizar_roleta(sorteado)
    # plt.pie(list(map(abs, populacao.keys())), list(map(abs, populacao.values())), normalize=True)
    # plt.title("Roleta")
    # plt.show()
    return sorteado


def crossover(populacao, pc, func):
    prox_pop = []
    for pai1 in populacao:
        if random.random() < pc:
            novofilho = copy.deepcopy(pai1)
            pai2 = roleta(populacao)
            novofilho[0], novofilho[1] = (pai1[0] + pai2[0]) / 2, (pai1[1] + pai2[1])/ 2
            novofilho[2] = func(novofilho[0], novofilho[1])  # Calcula fitness
            prox_pop.append(novofilho)
        else:
            prox_pop.append(pai1)

    return prox_pop


def mutacao(população, pm):
    # if (np.rand() > pm):
    #     for filho in população
    pass


def seleciona(populacao, tamanho_pop):
    nova_pop = []
    while len(nova_pop) < tamanho_pop:
        nova_pop.append(roleta(populacao))

    plt.pie([row[2] for row in populacao], labels=['{}, {}'.format(row[0], row[1]) for row in populacao])
    plt.title("População intermediária")
    plt.show()

    return nova_pop


def ordenar(populacao):
    dicionario_ordenado = {}  # Cria um dicionário vazio que será retornado
    fitness_ordenado = sorted(populacao, key=lambda x: x[2], reverse=True)  # Ordena o dicionário pelo x[1],
                                                                                    # isto é, pelo fitness associado
                                                                                    # a cada item
    return fitness_ordenado


def contabilizar_roleta(chave):
    if chave not in estatistica:
        estatistica[chave] = 1
    else:
        estatistica[chave] += 1


def objective(x,y):
    return 1 / ackley(x,y)


def ackley(x, y):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(0.5 * (cos(2 *
                                                                              pi * x) + cos(2 * pi * y))) + e + 20


if __name__ == '__main__':
    tamPop = 10
    pc = 0.9  # Probabilidade de Crossover
    pm = 0.1  # Probabilidade de mutação
    Ngera = 20  # Nro de gerações
    Nbits = 32  # Número de bits para cada variável
    Nvar = 2  # Nro de variáveis
    gera = 0  # Geração inicial
    lower = -32.768
    upper = 32.768
    # func =
    pop_final = algoritmo_genetico(Ngera, objective, pc, pm, tamPop, lower, upper)
    plt.pie([row[2] for row in pop_final], labels=['{}, {}'.format(row[0], row[1]) for row in pop_final])
    plt.title("População final")
    plt.show()
    ordenar(pop_final)
    print(str(pop_final))
