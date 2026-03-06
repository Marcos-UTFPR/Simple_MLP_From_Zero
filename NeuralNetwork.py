import os
import sys
import time
import random

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

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Exceções -------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class CustomException(Exception):
    def __init__(self, message=("CustomException")):
        # Call the base class constructor with the parameters it needs
        super(CustomException, self).__init__(message)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Funções auxiliares ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def step_function(x):
    if x < 0:
        return 0
    elif x >= 0:
        return 1
    
def writeLog(data, file_name):
    file_log = open(f"{file_name}.txt", "a") # Usado para salvar os pesos em um arquivo TXT
    file_log.write(data+'\n')
    file_log.close()
    
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
    def __init__(self, input_number, activation_function=step_function):
        self.input_number = input_number # Nome do tipo de peça dela
        self.weights = []
        for i in range(0,input_number): # Número de entradas esperadas nesse neurônio
            self.weights.append(random_value(-2,2))
        self.bias = random_value(-2,2)
        self.activation_function = activation_function

    def __iter__(self):
        return self

    # ---------------------------------------

    def output(self, inputs):
        assert len(inputs) == (self.input_number)
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
        pass

    # ---------------------------------------

    def load(self, new_weights):
        pass

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class Layer: # Camada de neurônios
    # Atributos: neuron_number, neurons, input_number (OBS: neuron_number também é o número de saídas que vai ser passada pra próxima)
    def __init__(self, neuron_number, input_number):
        self.input_number = input_number
        self.neuron_number = neuron_number
        self.neurons = []
        for i in range (0, neuron_number):
            self.neurons.append(perceptron(input_number))

    def __iter__(self):
        return self

    # ---------------------------------------

    def output(self,inputs):
        #print(f"Teste do assert: {len(inputs)} e {self.input_number}")
        assert len(inputs) == self.input_number
        final_outputs = []
        for neuron in self.neurons:
            final_outputs.append(neuron.output(inputs))
        return final_outputs # OBS: Tamanho desse vetor é o neuron_number
    
    # ---------------------------------------
    
    def open_black_box(self,layer_number):
        print(f"\tCamada {layer_number}")
        for i in range(0, len(self.neurons)):
            print(f"-Neurônio {i+1}:")
            self.neurons[i].open_black_box()

    # ---------------------------------------

    def save(self):
        pass

    # ---------------------------------------

    def load(self):
        pass

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class BuiltIn_MLP: # Multilayer Perceptron, com X camadas, todas com Y neurônios, gerando uma saída de tamanho Y (Y = neuron_number)
    # Atributos: input_number, layer_number, layers, neuron_number
    def __init__(self, input_number, layer_number, neuron_number):
        self.input_number = input_number
        self.layer_number = layer_number
        self.neuron_number = neuron_number
        self.layers = []
        for i in range(0, layer_number):
            self.layers.append(Layer(neuron_number, input_number))
            input_number = neuron_number

    # ---------------------------------------

    def output(self,inputs):
        assert len(inputs) == self.input_number
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

    def save(self):
        pass

    # ---------------------------------------

    def load(self):
        pass

    # ---------------------------------------
# ----------------------------------------------------------------------------------------------------

class Custom_MLP(): # Multilayer Perceptron customizável
    # Atributos: input_number, layer_number, layers
    def __init__(self, layers):
        self.layers = layers
        for layer in self.layers:
            #print(f"Teste do assert: {str(type(layer))} e {type(layer)}")
            if str(type(layer)) != "<class '__main__.Layer'>":
                raise TypeError

     # ---------------------------------------

    def output(self,inputs):
        assert len(inputs) == self.layers[0].input_number # Vê o tamanho da entrada da primeira camada
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

    def save(self):
        pass

    # ---------------------------------------

    def load(self):
        pass

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
                             Layer(3,8),
                             Layer(4,3),
                             Layer(4,4)   
                        ])
    input_values=[3,4,5,8,1,1,0,0]
    output_value = the_custom_network.output(input_values)
    print(f"Saída: {output_value}")
    #the_custom_network.open_black_box()

def test_save_and_load():
    pass

# ---------------------------------------

def main(): # Função principal
    clear() # Limpa o terminal
    # ---------------------------------------
    test_perceptron()
    # ---------------------------------------
    test_builtin_mlp()
    # ---------------------------------------
    test_custom_mlp()
    # ---------------------------------------
    test_save_and_load()

# ----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(YELLOW+"Programa encerrado via terminal..."+END)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Fim do código --------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------    
