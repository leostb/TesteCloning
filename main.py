import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import copy

from numpy.random import randint

from functions import ackley, dejong, gold, rastrigin, suums, printgraph, bump


class Cromossomo:
    def __init__(self, x, y, fitness):
        self.x = x
        self.y = y
        self.fitness = fitness


# TODO talvez representar o x e o y em uma única cadeia binária

estatistica = {}
fitness_medio = []
RELATORIO_DIR = 'relatorios'

functions = {
    gold: [-2, 2],
    suums: [-10, 10],
    dejong: [-2, 2],
    ackley: [-32.768, 32.768],
    bump: [0, 10],
    rastrigin: [-5, 5]
}


def algoritmo_genetico(numero_epocas, fitness_func, probabilidade_cross, probabilidade_mutacao, tamanho_pop, lower,
                       upper,
                       nbits):
    populacao = inicializar_população(tamanho_pop, lower, upper, nbits)
    populacao = calcular_fitness(populacao, fitness_func, lower, upper)
    populacao = ordenar(populacao)

    for i in range(0, numero_epocas):
        print("**** Geracao " + str(i) + "*****")
        imprimir_tabela_apenas(pop2real(populacao), "Geração " + str(i))
        popcross = crossover(populacao, probabilidade_cross, fitness_func)  # Reprodução
        mutacao(popcross, probabilidade_mutacao)  # TODO vai pegar os pais também, não deveria ter mutação nos pais
        # array_fitness = calcular_fitness(populacao, func)
        populacao += popcross
        populacao = calcular_fitness(populacao, fitness_func, lower, upper)
        populacao = ordenar(populacao)
        populacao = seleciona(populacao, tamanho_pop)  # Escolher um método tipo roleta
        calcular_fitness_medio(populacao)
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
    soma = 0
    intervalos = [0]
    for individuo in populacao:
        soma += individuo[2]
        intervalos.append(soma)

    ponto_sorteado = random.random() * soma
    for i in range(0, len(intervalos) - 1):
        if ponto_sorteado >= intervalos[i] and ponto_sorteado < intervalos[i + 1]:
            return copy.deepcopy(populacao[i])

    # contabilizar_roleta(sorteado)
    # plt.pie(list(map(abs, populacao.keys())), list(map(abs, populacao.values())), normalize=True)
    # plt.title("Roleta")
    # plt.show()
    # return sorteado


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
            # pai2 = populacao[randint(0, len(populacao))]
            pai2 = roleta(populacao)
            novofilho[0], novofilho[1] = cross(pai1[0], pai2[0]), cross(pai1[1], pai2[1])
            novofilho[2] = func(bin2real(novofilho[0], lower, upper),
                                bin2real(novofilho[1], lower, upper))  # Calcula fitness
            prox_pop.append(novofilho)
        else:
            prox_pop.append(pai1)

    return prox_pop


def mutacaoindividuo(individuo):
    for i in range(len(individuo)):
        if random.random() < pm:
            individuo[i] = 0 if individuo[i] == 1 else 1


def mutacao(população, pm):
    for individuo in população:
        mutacaoindividuo(individuo[0])
        mutacaoindividuo(individuo[1])


def seleciona(populacao, tamanho_pop):
    # nova_pop = []
    populacao = ordenar(populacao)
    nova_pop = populacao[0:10]
    for i in range(tamanho_pop):
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

@np.vectorize
def fitness(x, y):
    objective = func(x, y)
    if np.isnan(objective):
        return 0
    return 1 / (80 + objective)


def bin2real(xb, xmin, xmax):
    xr = 0
    N = len(xb)  # Numero de bits
    xb = xb[::-1]
    for i in range(0, N):
        xr += xb[i] * (2 ** i)

    return xmin + xr * (xmax - xmin) / (2 ** N - 1)  # x no intervalo [xmin,xmax]


def imprimir_tabela_apenas(data, titulo=None):
    n_rows = len(data)
    header = 'Individuo;x;y;Fitness'
    with open(os.path.join(PATH, "progressao" + '.csv'), 'a') as f:
        print(titulo, file=f)
        print(header, file=f)
        for i in range(n_rows):
            print('{};{};{};{}'.format(i, data[i][0], data[i][1], data[i][2]), file=f)


def imprimir_tabelae2d(data):
    n_rows = len(data)
    rows = ['Individuo {}'.format(i) for i in range(n_rows)]
    columns = ('x', 'y', 'Fitness')

    x, y = np.array(data)[:, 0], np.array(data)[:, 1]
    plt.scatter(x, y)

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=data,
                          rowLabels=rows,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Y")
    plt.savefig()


def pop2real(pop):
    pop_convertida = []
    for ind in pop:
        pop_convertida.append([bin2real(ind[0], lower, upper), bin2real(ind[1], lower, upper), ind[2]])
    return pop_convertida


def imprimir_parametros():
    parametros = {'isSeedOn': isSeedOn,
                  'seed': seed,
                  'seednp': seednp,
                  'tamPop': tamPop,
                  'pc': pc,  # Probabilidade de Crossover
                  'pm': pm,  # Probabilidade de mutação
                  'Ngera': Ngera,  # Nro de gerações
                  'Nbits': Nbits,  # Número de bits para cada variável
                  'Nvar': Nvar,  # Nro de variáveis
                  'gera': gera,  # Geração inicial
                  'func': func
                  }
    with open(os.path.join(PATH, 'parametros.txt'), 'w') as f:
        #print(input("Comentário:"), file=f)
        for k, v in parametros.items():
            print(str(k) + " = " + str(v), file=f)


def calcular_fitness_medio(populacao):
    # TODO talvez trocar variável global
    soma = 0.0
    for individuo in populacao:
        soma += individuo[2]
    media = soma / len(populacao)
    fitness_medio.append(media)
    with open(os.path.join(PATH, "progressao" + '.csv'), 'a') as f:
        print("Fitness Médio: " + str(media) + "\n", file=f)


def imprimir_fitness_medio():
    plt.plot(fitness_medio)
    plt.savefig(os.path.join(PATH + "/fitnessmedio"))


if __name__ == '__main__':
    isTemp = False  # Salvar em pasta temporária
    isSeedOn = True
    seed = 2
    seednp = 1
    if isSeedOn:
        random.seed(seed)
        np.random.seed(seednp)

    if isTemp:
        PATH = os.path.join(RELATORIO_DIR, "temp", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(PATH)
    else:
        PATH = os.path.join(RELATORIO_DIR, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(PATH)

    tamPop = 30
    pc = 0.8  # Probabilidade de Crossover
    pm = 0.01  # Probabilidade de mutação
    Ngera = 500  # Nro de gerações
    Nbits = 32  # Número de bits para cada variável
    Nvar = 2  # Nro de variáveis
    gera = 0  # Geração inicial

    func = rastrigin

    lower = functions[func][0]
    upper = functions[func][1]

    imprimir_parametros()

    printgraph(lower, upper, 100, fitness)
    printgraph(lower, upper, 100, func)

    pop_final = algoritmo_genetico(Ngera, fitness, pc, pm, tamPop, lower, upper, Nbits)
    imprimir_fitness_medio()
