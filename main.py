import random

import matplotlib.pyplot as plt
import copy

from numpy.random import randint

from bumpfunction import ackley, dejong, gold, rastrigin, suums


class Cromossomo:
    def __init__(self, x, y, fitness):
        self.x = x
        self.y = y
        self.fitness = fitness


# TODO incluir bounds
# TODO talvez representar o x e o y em uma única cadeia binária

estatistica = {}


def algoritmo_genetico(numero_epocas, func, probabilidade_cross, probabilidade_mutacao, tamanho_pop, lower, upper,
                       nbits):
    populacao = inicializar_população(tamanho_pop, lower, upper)
    populacao = calcular_fitness(populacao, func, lower, upper)
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
        populacao = calcular_fitness(populacao, func, lower, upper)
    return populacao


def inicializar_população(tamanho_pop, lower, upper, n_bits=32):
    # Retorna um array de pares x,y
    # Poderia já calcular o fitness aqui, mas por questão de didatica vou calcular depoist
    populacao = []
    for i in range(tamanho_pop):
        populacao.append([randint(0, 2, size=n_bits).tolist(), randint(0, 2, size=n_bits).tolist(), 0])
    return populacao


def calcular_fitness(populacao, func, lower, upper):
    for item in populacao:
        item[2] = func(bin2real(item[0], lower, upper), bin2real(item[1], lower,
                                                                 upper))  # TODO Refatorar, ao invés de trabalhar com lista, poderia fazer uma classe
    # TODO testei e não precisava retornar, ele já alterou o objeto original
    return populacao


def roleta(populacao):
    # TODO melhorar ficou muito tosco hausashu muita conversão de tipo para fazer isso
    # TODO talvez tenha um método que receba apenas o dicionario
    sorteado = random.choices(populacao, [row[2] for row in populacao]).pop()
    # contabilizar_roleta(sorteado)
    # plt.pie(list(map(abs, populacao.keys())), list(map(abs, populacao.values())), normalize=True)
    # plt.title("Roleta")
    # plt.show()
    return sorteado


def cross(individuo1, individuo2):
    n = len(individuo1)
    ponto_corte = randint(1,
                          n)  # Não estou permitindo que um filho seja completamente igual a nenhum dos pais, isto é, não pode dar 0 nem N
    filho = []
    for i in range(n):
        if i < ponto_corte:
            filho.append(individuo1[i])
        else:
            filho.append(individuo2[i])
    return filho


def crossover(populacao, pc, func):
    prox_pop = []
    for pai1 in populacao:
        if random.random() < pc:
            novofilho = copy.deepcopy(pai1)
            pai2 = roleta(populacao)  # TODO alterar de roleta para randômico talvez
            novofilho[0], novofilho[1] = cross(pai1[0], pai2[0]), cross(pai1[1], pai2[1])
            novofilho[2] = func(bin2real(novofilho[0], lower, upper),
                                bin2real(novofilho[1], lower, upper))  # Calcula fitness
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

    # plt.pie([row[2] for row in populacao], labels=['{}, {}'.format(row[0], row[1]) for row in populacao])
    # plt.title("População intermediária")
    # plt.show()

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


def objective(x, y):
    return 1 / func(x, y)


def bin2real(xb, xmin, xmax):
    xr = 0
    N = len(xb)  # Numero de bits
    xb = xb[::-1]
    for i in range(0, N):
        xr += xb[i] * (2 ** i)

    return xmin + xr * (xmax - xmin) / (2 ** N - 1)  # x no intervalo [xmin,xmax]


if __name__ == '__main__':
    tamPop = 10
    pc = 0.9  # Probabilidade de Crossover
    pm = 0.1  # Probabilidade de mutação
    Ngera = 100  # Nro de gerações
    Nbits = 32  # Número de bits para cada variável
    Nvar = 2  # Nro de variáveis
    gera = 0  # Geração inicial

    func = ackley
    functions = {
        gold: [-2, 2],
        suums: [-10, 10],
        dejong: [-2, 2],
        ackley: [-32.768, 32.768],
        rastrigin: [-5,5]
    }

    lower = functions[func][0]
    upper = functions[func][1]

    pop_final = algoritmo_genetico(Ngera, objective, pc, pm, tamPop, lower, upper, Nbits)
    plt.pie([row[2] for row in pop_final], labels=['{}, {}'.format(row[0], row[1]) for row in pop_final])
    plt.title("População final")
    plt.show()
    pop_final_convertida = []
    for ind in pop_final:
        pop_final_convertida.append([bin2real(ind[0], lower, upper), bin2real(ind[1], lower, upper), ind[2]])
    print(str(pop_final_convertida))
