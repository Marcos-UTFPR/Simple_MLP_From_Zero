# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Versão experimental feita com ajuda do Claude para entender onde eu estava errando na branch dev ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------
import os
import sys
import time
import random
import math

import random as rand_module
from collections import deque

GAMMA = 0.9             # Fator de desconto
EPSILON_START = 1.0     # Exploração inicial (100% aleatório)
EPSILON_MIN = 0.01      # Exploração mínima
EPSILON_DECAY = 0.995   # Decay por episódio
MEMORY_SIZE = 10000     # Tamanho do replay buffer
BATCH_SIZE = 64         # Amostras por treino

class DQN_Agent:
    def __init__(self, network):
        # network é sua Custom_MLP já instanciada
        self.network = network
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)  # Replay buffer

    # --------------------------------------------------

    def escolher_acao(self, estado):
        # Epsilon-greedy: no início age aleatório, aos poucos usa a rede
        if rand_module.random() < self.epsilon:
            return rand_module.randint(0, 3)  # 0=cima, 1=baixo, 2=esq, 3=dir
        valores_q = self.network.output(estado)
        return valores_q.index(max(valores_q))  # Ação de maior valor Q

    # --------------------------------------------------

    def memorizar(self, estado, acao, recompensa, proximo_estado, morreu):
        # Guarda a experiência para replay
        self.memory.append((estado, acao, recompensa, proximo_estado, morreu))

    # --------------------------------------------------

    def treinar_passo_unico(self, estado, acao, recompensa, proximo_estado, morreu):
        # Treino imediato após cada ação (sem replay)
        valores_q_atual = self.network.output(estado)

        if morreu:
            q_alvo = recompensa
        else:
            valores_q_proximo = self.network.output(proximo_estado)
            q_alvo = recompensa + GAMMA * max(valores_q_proximo)

        # Só altera o Q da ação tomada — as outras ficam iguais
        expected = list(valores_q_atual)
        expected[acao] = q_alvo

        self.network.backward(estado, expected, valores_q_atual)

    # --------------------------------------------------

    def treinar_replay(self):
        # Experience replay: treina num batch aleatório da memória
        # Estabiliza o treino quebrando correlação entre experiências consecutivas
        if len(self.memory) < BATCH_SIZE:
            return  # Aguarda memória encher

        batch = rand_module.sample(list(self.memory), BATCH_SIZE)

        for estado, acao, recompensa, proximo_estado, morreu in batch:
            valores_q_atual = self.network.output(estado)

            if morreu:
                q_alvo = recompensa
            else:
                valores_q_proximo = self.network.output(proximo_estado)
                q_alvo = recompensa + GAMMA * max(valores_q_proximo)

            expected = list(valores_q_atual)
            expected[acao] = q_alvo

            self.network.backward(estado, expected, valores_q_atual)

    # --------------------------------------------------

    def decair_epsilon(self):
        # Chama isso ao fim de cada episódio (cada partida)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Constantes -----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Só vão ser alteradas dentro do desenvolvimento

# Cores
PURPLE = '\033[95m'
CYAN = '\033[96m'
DARKCYAN = '\033[36m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m'

MAX_WEIGHT = 1  # Valor máximo dos pesos e bias (Default: 2)
MIN_WEIGHT = -1 # Valor mínimo dos pesos e bias (Default: -2)

DEFAULT_LEARNING_RATE = 0.01 # Valor padrão da Taxa de Aprendizado (Default: 0.1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Exceções -------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class InvalidInputSizeException(Exception):
    def __init__(self, message=("Invalid Input Size!")):
        # Call the base class constructor with the parameters it needs
        super(InvalidInputSizeException, self).__init__(message)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Funções de Ativação --------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def step_function(x): # Função de ativação básica
    if x < 0:
        return 0
    elif x >= 0:
        return 1

def relu(x): # Função de ativação ReLU
    return max(0, x)

def sig(x):
    return 1/(1 + math.exp(-x))

DEFAULT_ACTIVATION_FUNCTION = sig # (Default: sig)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Funções de erro ------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def basic_error(expected_values, predicted_values): # Função de cálculo de erro básica
    if len(expected_values) != len(predicted_values):
        #print(f"Teste do assert: {len(expected_values)} e {len(predicted_values)}")
        raise InvalidInputSizeException
    errors = []
    for i in range(0, len(expected_values)):
        errors.append(predicted_values[i]-expected_values[i])
    return errors

def mse(expected_values, predicted_values): # Mean Squared Error (MSE)
    errors = basic_error(expected_values, predicted_values)
    for i in range(0, len(errors)):
        errors[i] = errors[i] ** 2
    return errors

DEFAULT_ERROR_FUNCTION = mse # (Default: mse)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Funções auxiliares ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------
    
def writeLog(data, file_name):
    data = str(data)
    file_log = open(f"{file_name}.txt", "w") # Usado para salvar os pesos em um arquivo TXT
    file_log.write(data+'\n')
    file_log.close()

def getWeightsFromLog(file_name): # Tem que ser um txt
    file_log = open(f"{file_name}.txt")
    data = file_log.read()
    file_log.close()
    return eval(data)
    
def random_value(min, max):
    return round(random.uniform(min,max), 2) # Arredonda o valor para 2 decimais apenas

def typewriterPrint(message): # Um print lento com efeito de "máquina de escrever"
    for x in message:
        print(x, end='')
        sys.stdout.flush()
        time.sleep(0.1)
    print('\n', end='') # Pulando linha

def clear(): # Limpa o terminal
    os.system('cls' if os.name == 'nt' else 'clear')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Outros ---------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def doNothing():
    pass

def doNothingForApproximately(seconds): # Um time.sleep() bem piorado
    # 1 ciclo quase é 1 segundo exato
    for i in range(seconds):
        i = 0
        while i < 16706910:
            pass
            i += 1

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Classes principais ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class perceptron:
    # Atributos: input_number, weights, bias, activation_function
    def __init__(self, input_number, activation_function=DEFAULT_ACTIVATION_FUNCTION):
        self.input_number = input_number # Nome do tipo de peça dela
        self.weights = []
        for i in range(0,input_number): # Número de entradas esperadas nesse neurônio
            self.weights.append(random_value(MIN_WEIGHT,MAX_WEIGHT))
        self.bias = random_value(MIN_WEIGHT,MAX_WEIGHT)
        self.activation_function = activation_function

    def __iter__(self):
        return self

    # ---------------------------------------

    def output(self, inputs):
        if len(inputs) != (self.input_number):
            raise InvalidInputSizeException
        outputs = []
        final_output = 0
        for i in range(0,self.input_number): # Número de entradas esperadas nesse neurônio
            outputs.append(inputs[i]*self.weights[i])
        for i in outputs:
            final_output += i
        final_output += self.bias
        final_output = self.activation_function(final_output)
        return final_output

    # ---------------------------------------

    def open_black_box(self):
        for i in range(0,len(self.weights)): # Número de entradas esperadas nesse neurônio
            print(f"Peso {i+1}: {self.weights[i]}")
        print(f"Viés: {self.bias}")

    # ---------------------------------------

    def save(self):
        weights = self.weights + [self.bias] # Pesos + Viés
        return weights # Só retorna o valor, a classe MLP é responsável por salvar em arquivo

    # ---------------------------------------

    def load(self, new_weights):
        if len(new_weights) != len(self.weights)+1: # Pesos + Viés
            #print(f"Teste do assert: {len(new_weights)} e {len(self.weights)+1}")
            raise InvalidInputSizeException
        for i in range(0,len(self.weights)):
            self.weights[i] = new_weights[i]
        self.bias = new_weights[-1] # Em teoria, o viés é o último valor entre os pesos

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class Layer: # Camada de neurônios
    # Atributos: neuron_number, neurons, input_number (OBS: neuron_number também é o número de saídas que vai ser passada pra próxima)
    def __init__(self, neuron_number, input_number):
        self.input_number = input_number
        self.neuron_number = neuron_number
        self.neurons = []
        for i in range (0, neuron_number):
            self.neurons.append(perceptron(input_number)) # OBS: Todos os neurônios tem o mesmo tamanho de entrada (densamente conectada)

    def __iter__(self):
        return self

    # ---------------------------------------

    def output(self,inputs):
        #print(f"Teste do assert: {len(inputs)} e {self.input_number}")
        if len(inputs) != self.input_number:
            raise InvalidInputSizeException
        final_outputs = []
        for neuron in self.neurons:
            final_outputs.append(neuron.output(inputs))
        return final_outputs # OBS: Tamanho desse vetor é o neuron_number
    
    # ---------------------------------------
    
    def open_black_box(self,layer_number):
        print(f"\tCamada {layer_number+1}")
        for i in range(0, len(self.neurons)):
            print(f"-Neurônio {i+1}:")
            self.neurons[i].open_black_box()

    # ---------------------------------------

    def save(self):
        weights = []
        for neuron in self.neurons:
            weights += neuron.save() # Retorna os pesos
        return weights # Retorna os pesos dos neurônios em ordem

    # ---------------------------------------

    def load(self, new_weights):
        if len(new_weights) != ((self.input_number*self.neuron_number)+self.neuron_number): # Pesos totais = Entrada * N° de neurônios
            #print(f"Teste do assert: {len(new_weights)} e {(self.input_number*self.neuron_number)+self.neuron_number}")
            raise InvalidInputSizeException
        x=0 # Auxiliar
        for i in range(0,self.neuron_number):
            new_parameters = new_weights[x:x+self.input_number] # Separando os pesos para cada neurônio
            new_parameters += [new_weights[x+self.input_number]] # O último depois de cada peso é pra ser o viés
            x+=self.input_number+1 # Recomeça a partir do viés
            self.neurons[i].load(new_parameters)

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class BuiltIn_MLP: # Multilayer Perceptron, com X camadas, todas com Y neurônios, gerando uma saída de tamanho Y (Y = neuron_number)
    # Atributos: input_number, layer_number, layers, neuron_number
    def __init__(self, input_number, layer_number, neuron_number, learning_rate = DEFAULT_LEARNING_RATE, error_function = DEFAULT_ERROR_FUNCTION):
        self.input_number = input_number
        self.layer_number = layer_number
        self.neuron_number = neuron_number
        self.layers = []
        self.learning_rate = learning_rate
        self.error_function = error_function
        for i in range(0, layer_number):
            self.layers.append(Layer(neuron_number, input_number))
            input_number = neuron_number

    # ---------------------------------------

    def output(self,inputs): # Forward Pass
        if len(inputs) != self.input_number:
            raise InvalidInputSizeException
        final_outputs = None
        for layer in self.layers:
            inputs = layer.output(inputs) # Saída de uma camada é entrada da próxima
            final_outputs = inputs
        return final_outputs # OBS: Tamanho desse vetor é o neuron_number
    
    # ---------------------------------------
    
    def open_black_box(self):
        print(f"\t\tMultilayer Perceptron")
        for i in range(0,len(self.layers)):
            self.layers[i].open_black_box(i)

    # ---------------------------------------

    def architecture_info(self):
        for i in range(0,len(self.layers)):
            print(f"Camada {i+1}: {self.layers[i].neuron_number} Neurônios - {self.layers[i].input_number} Inputs") # Todas tem o mesmo tamanho aqui na verdade

    # ---------------------------------------

    def save(self):
        weights = []
        for layer in self.layers:
            weights += layer.save() # Retorna os pesos
        return weights # Retorna os pesos dos neurônios de cada camada em ordem

    # ---------------------------------------

    def load(self, new_weights):
        x=0 # Auxiliar
        for i in range(0,len(self.layers)):
            new_parameters = new_weights[x:x+(self.layers[i].input_number*self.layers[i].neuron_number)+self.layers[i].neuron_number] # Separando os pesos para cada neurônio
            x+=(self.layers[i].input_number*self.layers[i].neuron_number)+self.layers[i].neuron_number # neuron_number também é a quantidade de viéses que tem na camada
            self.layers[i].load(new_parameters)

    # ---------------------------------------

    def backward(self, inputs, expected_values, predicted_values):
        if len(expected_values) != len(predicted_values):
            raise InvalidInputSizeException

        # Forward pass completo, guardando saídas separadas por camada
        layer_outputs = []
        current = inputs
        for layer in self.layers:
            current = layer.output(current)
            layer_outputs.append(list(current))
        # layer_outputs[0] = saídas da camada 1, layer_outputs[1] = camada 2, etc.
        # layer_outputs[-1] = saídas da última camada = predicted_values

        # Entrada de cada camada:
        # camada 0 recebe os inputs originais
        # camada i recebe layer_outputs[i-1]
        layer_inputs = [list(inputs)] + layer_outputs[:-1]

        # -------------------------------------------------------
        # Calculando os deltas da camada de saída
        # delta = derivada_da_ativacao * (previsto - esperado)
        # derivada da sigmoid = saida * (1 - saida)
        # -------------------------------------------------------
        deltas = [None] * len(self.layers)

        output_deltas = []
        for j in range(self.layers[-1].neuron_number):
            saida = layer_outputs[-1][j]
            erro = saida - expected_values[j]  # previsto - esperado
            delta = saida * (1 - saida) * erro
            output_deltas.append(delta)
        deltas[-1] = output_deltas

        # -------------------------------------------------------
        # Calculando os deltas das camadas escondidas, de trás pra frente
        # delta_i = derivada_da_ativacao_i * soma(peso_ij * delta_j)
        # onde j percorre todos os neurônios da camada SEGUINTE
        # e peso_ij é o peso da conexão de i para j (guardado no neurônio j)
        # -------------------------------------------------------
        for i in range(len(self.layers) - 2, -1, -1):  # da penúltima até a primeira
            layer_deltas = []
            next_layer = self.layers[i + 1]
            next_deltas = deltas[i + 1]

            for j in range(self.layers[i].neuron_number):
                saida = layer_outputs[i][j]

                # Soma: para cada neurônio da camada seguinte,
                # pega o peso que conecta o neurônio j (desta camada) a ele,
                # e multiplica pelo delta daquele neurônio
                soma = 0
                for k in range(next_layer.neuron_number):
                    peso_j_para_k = next_layer.neurons[k].weights[j]
                    soma += peso_j_para_k * next_deltas[k]

                delta = saida * (1 - saida) * soma
                layer_deltas.append(delta)

            deltas[i] = layer_deltas

        # -------------------------------------------------------
        # Atualizando os pesos
        # novo_peso = peso_atual - learning_rate * delta_destino * saida_origem
        # novo_bias  = bias_atual - learning_rate * delta_destino
        # -------------------------------------------------------
        for i in range(len(self.layers)):
            entradas_desta_camada = layer_inputs[i]
            for j in range(self.layers[i].neuron_number):
                delta_j = deltas[i][j]
                for k in range(len(self.layers[i].neurons[j].weights)):
                    saida_origem = entradas_desta_camada[k]
                    self.layers[i].neurons[j].weights[k] -= self.learning_rate * delta_j * saida_origem
                # Bias: origem imaginária com saída 1, então não multiplica por nada
                self.layers[i].neurons[j].bias -= self.learning_rate * delta_j

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class Custom_MLP(): # Multilayer Perceptron customizável
    # Atributos: input_number, layer_number, layers
    def __init__(self, layers, learning_rate = DEFAULT_LEARNING_RATE, error_function = DEFAULT_ERROR_FUNCTION):
        self.layers = layers
        self.learning_rate = learning_rate
        self.error_function = error_function
        for layer in self.layers:
            #print(f"Teste do assert: {str(type(layer))} e {type(layer)}")
            if not isinstance(layer, Layer):
                raise InvalidInputSizeException

     # ---------------------------------------

    def output(self,inputs): # Forward Pass
        if len(inputs) != self.layers[0].input_number: # Vê o tamanho da entrada da primeira camada
            raise InvalidInputSizeException
        final_outputs = None
        for layer in self.layers:
            inputs = layer.output(inputs) # Saída de uma camada é entrada da próxima
            final_outputs = inputs
        return final_outputs # OBS: Tamanho desse vetor é o neuron_number da última camada
    
    # ---------------------------------------

    def output_training(self,inputs): # Output utilizado para propósitos de treinamento, retorna as saídas de todos os neurônio
        if len(inputs) != self.layers[0].input_number: # Vê o tamanho da entrada da primeira camada
            raise InvalidInputSizeException
        final_outputs = []
        for layer in self.layers:
            inputs = layer.output(inputs) # Saída de uma camada é entrada da próxima 
            for input in inputs:
                final_outputs.append(input) # Salvando todos os pesos
        return final_outputs # OBS: Tamanho desse vetor é o total de neurônios da rede
    
    # ---------------------------------------
    
    def open_black_box(self):
        print(f"\t\tMultilayer Perceptron")
        for i in range(0,len(self.layers)):
            self.layers[i].open_black_box(i)

    # ---------------------------------------

    def architecture_info(self):
        for i in range(0,len(self.layers)):
            print(f"Camada {i+1}: {self.layers[i].neuron_number} Neurônios - {self.layers[i].input_number} Inputs")

    # ---------------------------------------

    def save(self):
        weights = []
        for layer in self.layers:
            weights += layer.save() # Retorna os pesos
        return weights # Retorna os pesos dos neurônios de cada camada em ordem

    # ---------------------------------------

    def load(self, new_weights):
        x=0 # Auxiliar
        for i in range(0,len(self.layers)):
            new_parameters = new_weights[x:x+(self.layers[i].input_number*self.layers[i].neuron_number)+self.layers[i].neuron_number] # Separando os pesos para cada neurônio
            x+=(self.layers[i].input_number*self.layers[i].neuron_number)+self.layers[i].neuron_number # neuron_number também é a quantidade de viéses que tem na camada
            self.layers[i].load(new_parameters)

    # ---------------------------------------

    def backward(self, inputs, expected_values, predicted_values):
        if len(expected_values) != len(predicted_values):
            raise InvalidInputSizeException

        # Forward pass completo, guardando saídas separadas por camada
        layer_outputs = []
        current = inputs
        for layer in self.layers:
            current = layer.output(current)
            layer_outputs.append(list(current))
        # layer_outputs[0] = saídas da camada 1, layer_outputs[1] = camada 2, etc.
        # layer_outputs[-1] = saídas da última camada = predicted_values

        # Entrada de cada camada:
        # camada 0 recebe os inputs originais
        # camada i recebe layer_outputs[i-1]
        layer_inputs = [list(inputs)] + layer_outputs[:-1]

        # -------------------------------------------------------
        # Calculando os deltas da camada de saída
        # delta = derivada_da_ativacao * (previsto - esperado)
        # derivada da sigmoid = saida * (1 - saida)
        # -------------------------------------------------------
        deltas = [None] * len(self.layers)

        output_deltas = []
        for j in range(self.layers[-1].neuron_number):
            saida = layer_outputs[-1][j]
            erro = saida - expected_values[j]  # previsto - esperado
            delta = saida * (1 - saida) * erro
            output_deltas.append(delta)
        deltas[-1] = output_deltas

        # -------------------------------------------------------
        # Calculando os deltas das camadas escondidas, de trás pra frente
        # delta_i = derivada_da_ativacao_i * soma(peso_ij * delta_j)
        # onde j percorre todos os neurônios da camada SEGUINTE
        # e peso_ij é o peso da conexão de i para j (guardado no neurônio j)
        # -------------------------------------------------------
        for i in range(len(self.layers) - 2, -1, -1):  # da penúltima até a primeira
            layer_deltas = []
            next_layer = self.layers[i + 1]
            next_deltas = deltas[i + 1]

            for j in range(self.layers[i].neuron_number):
                saida = layer_outputs[i][j]

                # Soma: para cada neurônio da camada seguinte,
                # pega o peso que conecta o neurônio j (desta camada) a ele,
                # e multiplica pelo delta daquele neurônio
                soma = 0
                for k in range(next_layer.neuron_number):
                    peso_j_para_k = next_layer.neurons[k].weights[j]
                    soma += peso_j_para_k * next_deltas[k]

                delta = saida * (1 - saida) * soma
                layer_deltas.append(delta)

            deltas[i] = layer_deltas

        # -------------------------------------------------------
        # Atualizando os pesos
        # novo_peso = peso_atual - learning_rate * delta_destino * saida_origem
        # novo_bias  = bias_atual - learning_rate * delta_destino
        # -------------------------------------------------------
        for i in range(len(self.layers)):
            entradas_desta_camada = layer_inputs[i]
            for j in range(self.layers[i].neuron_number):
                delta_j = deltas[i][j]
                for k in range(len(self.layers[i].neurons[j].weights)):
                    saida_origem = entradas_desta_camada[k]
                    self.layers[i].neurons[j].weights[k] -= self.learning_rate * delta_j * saida_origem
                # Bias: origem imaginária com saída 1, então não multiplica por nada
                self.layers[i].neurons[j].bias -= self.learning_rate * delta_j

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Seção principal do código --------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def test_perceptron():
    print("\\---------- Teste de Perceptron -----------/")
    the_first_neuron = perceptron(8)
    input_values=[3,4,5,8,1,1,0,0] # Snake: Posição da Head, Posição da Fruit, Se cada uma das 4 direções está ocupada (1) ou não (0)
    output_value = the_first_neuron.output(input_values)
    print(f"Saída: {output_value}")
    the_first_neuron.open_black_box()

def test_builtin_mlp():
    print("\\---------- Teste de MLP -----------/")
    the_first_network = BuiltIn_MLP(8, 3, 4) # Snake: Saída final são 4 valores, 1 pra cada direção
    input_values=[3,4,5,8,1,1,0,0]
    output_value = the_first_network.output(input_values)
    print(f"Saída: {output_value}")
    #the_first_network.open_black_box()

def test_custom_mlp():
    print("\\---------- Teste de MLP Customizada -----------/")
    the_custom_network = Custom_MLP([   # Três camadas com tamanhos diferentes
                             Layer(3,8), # OBS: (número de neurônios, número de entradas)
                             Layer(4,3),
                             Layer(4,4)  # OBS: número de neurônios da última camada vai ser o tamanho da saída 
                        ])
    input_values=[3,4,5,8,1,1,0,0]
    output_value = the_custom_network.output(input_values)
    print(f"Saída: {output_value}")
    #the_custom_network.open_black_box()

def test_save_and_load_with_perceptron():
    print("Primeiro neurônio...")
    the_saved_neuron = perceptron(3)
    input_values=[3,4,5] # Snake: Posição da Head, Posição da Fruit, Se cada uma das 4 direções está ocupada (1) ou não (0)
    output_value = the_saved_neuron.output(input_values)
    the_saved_neuron.open_black_box()
    print(f"Saída 1: {output_value}")
    weights = the_saved_neuron.save()
    print(f"Pesos: {weights}")
    # -----
    print("Segundo neurônio...")
    the_second_neuron = perceptron(3)
    the_second_neuron.open_black_box()
    output_value = the_second_neuron.output(input_values)
    print(f"Saída 2: {output_value}")
    print("Carregando pesos do neurônio anterior...")
    the_second_neuron.load(weights)
    the_second_neuron.open_black_box()
    output_value = the_second_neuron.output(input_values)
    print(f"Saída 3: {output_value}")

def test_save_and_load_with_layer():
    test_layer = Layer(3,8)
    weights = test_layer.save()
    print(f"Pesos da Camada 0: {weights}")
    weights = [0.68, 1.28, 0.98, 0.86, 1.32, 1.17, 0.46, -0.35, 
               0.96, 0.65, 0.07, 0.82, 0.36, -0.58, -0.23, 0.64, 
               -0.22, 1.39, 0.93, -0.13, 0.47, 0.04, -1.74, -0.67] # Pesos de testes para 3 neurônios com 8 entradas
    test_layer.load(weights)
    print(f"Novos pesos da Camada 0: {weights}")
    the_layers_network = Custom_MLP([   # Três camadas com tamanhos diferentes
                             test_layer, # OBS: (número de neurônios, número de entradas)
                             Layer(4,3),
                             Layer(4,4)   
                        ])
    the_layers_network.architecture_info()
    the_layers_network.open_black_box()

def test_save_and_load_with_mlp():
    # OBS: Os 99 é para serem os viéses
    weights = [0.68, 1.28, 99, 0.98, 0.86, 99,  1.32, 1.17, 99,
               0.46, -0.35, 0.96, 99, 0.65, 0.07, 0.82, 99, 0.36, -0.58, -0.23, 99] # Pesos de testes para as duas camadas
    the_saved_network = Custom_MLP([   # Três camadas com tamanhos diferentes
                             Layer(3,2), # OBS: (número de neurônios, número de entradas)
                             Layer(3,3)   
                        ])
    pre_weights = the_saved_network.save()
    print(f"Pesos da Rede: {pre_weights}")
    the_saved_network.architecture_info()
    the_saved_network.open_black_box()
    # ----
    the_saved_network.load(weights)
    pre_weights = the_saved_network.save()
    print(f"Novos pesos da Rede: {pre_weights}")
    the_saved_network.architecture_info()
    the_saved_network.open_black_box()

def test_save_and_load_from_file():
    the_saved_network = Custom_MLP([   # Três camadas com tamanhos diferentes
                             Layer(3,2), # OBS: (número de neurônios, número de entradas)
                             Layer(3,3)   
                        ])
    try:
        pre_weights = getWeightsFromLog("test_weights") # Pegando os pesos
        print(f"Pesos antigos: {pre_weights}")
    except FileNotFoundError:
        pass
    else:
        the_saved_network.load(pre_weights) # Carregando os pesos
    weights = the_saved_network.save()
    #writeLog(weights, "test_weights") # Salvando pesos
    the_saved_network.open_black_box()

def test_training():
    print("\\---------- Teste de Treinamento -----------/")
    the_custom_network = Custom_MLP([   # Três camadas com tamanhos diferentes
                             Layer(3,8), # OBS: (número de neurônios, número de entradas)
                             Layer(4,3),
                             Layer(4,4)  # OBS: número de neurônios da última camada vai ser o tamanho da saída 
                        ])
    input_values=[3,4,5,8,1,1,0,0]
    expected_output = [0.25, 1, 0.73, 0]
    output_value = the_custom_network.output(input_values)
    # Exemplo de saídas = [0, 30.775000000000002, 0, 0]
    print(f"Saída predita: {output_value}")
    print(f"Saída esperada: {expected_output}")
    print(f"Erros: {mse(expected_output, output_value)}")
    print("-------- Badpropagation --------")
    for i in range(0,10000):
        the_custom_network.backward(input_values, expected_output, output_value) # inputs, expected_values, predicted_values
        output_value = the_custom_network.output(input_values)
    output_value = the_custom_network.output(input_values)
    # Exemplo de saídas = [0, 30.775000000000002, 0, 0]
    print(f"Saída predita 2: {output_value}")
    print(f"Saída esperada 2: {expected_output}")
    print(f"Erros 2: {mse(expected_output, output_value)}")

# ---------------------------------------

def main(): # Função principal
    clear() # Limpa o terminal
    # ---------------------------------------
    #test_perceptron()
    # ---------------------------------------
    #test_builtin_mlp()
    # ---------------------------------------
    #test_custom_mlp()
    # ---------------------------------------
    #test_save_and_load_with_perceptron()
    # ---------------------------------------
    #test_save_and_load_with_layer()
    # ---------------------------------------
    #test_save_and_load_with_mlp()
    # ---------------------------------------
    #test_save_and_load_from_file()
    # ---------------------------------------
    test_training()

# ----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(YELLOW+"Programa encerrado via terminal..."+END)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Fim do código --------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------    